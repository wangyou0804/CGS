import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import (
    BaseDataSets,
    TwoStreamBatchSampler,
    OVRWeakStrongAugment
)
from networks.net_factory import net_factory
from utils import losses, ramps, util
from utils.losses import cross_entropy_masked
from val_2D import test_single_volume_ovr

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str,
                    default="ACDC", help="Name of Experiment")
parser.add_argument(
    "--exp", type=str, default="ACDC/CGS", help="experiment_name")
parser.add_argument("--model", type=str, default="OVRUNet", help="model_name")
parser.add_argument("--max_iterations", type=int,
                    default=30000, help="maximum epoch number to train")
parser.add_argument("--batch_size", type=int, default=24,
                    help="batch_size per gpu")
parser.add_argument("--deterministic", type=int, default=1,
                    help="whether use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01,
                    help="segmentation network learning rate")
parser.add_argument("--patch_size", type=list,
                    default=[256, 256], help="patch size of network input")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4,
                    help="output channel of network")
parser.add_argument("--load", default=False,
                    action="store_true", help="restore previous checkpoint")
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.95,
    help="confidence threshold for using pseudo-labels",
)
parser.add_argument(
    "--cut_p",
    type=float,
    default=1.0,
    help="cutmix probility",
)
parser.add_argument("--labeled_bs", type=int, default=12,
                    help="labeled_batch_size per gpu")
parser.add_argument("--labeled_num", type=int, default=3, help="labeled data")
parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
parser.add_argument("--consistency_type", type=str,
                    default="mse", help="consistency_type")
parser.add_argument("--consistency", type=float,
                    default=0.1, help="consistency")
parser.add_argument("--consistency_rampup", type=float,
                    default=200.0, help="consistency_rampup")
parser.add_argument("--gpu", type=str, default="0",
                    help="GPU ids separated by comma")
args = parser.parse_args()

def patients_to_slices(dataset, patiens_num):
    print(dataset)
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {
            "3": 68,
            "7": 136
        }

    elif "SegTHOR" in dataset:
        ref_dict = {
            "3": 364,
            "6": 735,
        }
    elif "CHAOS" in dataset:
        ref_dict = {
            "2": 72,
            "5": 162,
        }
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step, buffer=False):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    if buffer:
        for (name, buffer), (ema_name, ema_buffer) in zip(
            model.named_buffers(), ema_model.named_buffers()
        ):
            if name == ema_name and "running" in name:
                ema_buffer.mul_(alpha).add_(buffer, alpha=1 - alpha)


def random_mask(bs, img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.ones(bs, img_size, img_size)
    if random.random() > p:
        return mask.cuda()

    for i in range(bs):
        size = np.random.uniform(size_min, size_max) * img_size * img_size
        while True:
            ratio = np.random.uniform(ratio_1, ratio_2)
            cutmix_w = int(np.sqrt(size / ratio))
            cutmix_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_size)
            y = np.random.randint(0, img_size)

            if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
                break
        mask[i: y:y + cutmix_h, x:x + cutmix_w] = 0
    return mask.cuda()


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def generate_ovr_pseudu_label(pseudo, args):
        pseudo_cpu = pseudo.cpu()
        ovr_label = []
        for i in range(args.num_classes - 1):
            # set label to 1 if it is the current class
            pix_2 = list(range(1, args.num_classes))
            if i + 1 in pix_2:
                pix_2.remove(i+1)
            cur_label = np.zeros_like(pseudo_cpu)
            cur_label[pseudo_cpu == i + 1] = 1
            for pix in pix_2:
                cur_label[pseudo_cpu == pix] = 2
            ovr_label.append(
                torch.from_numpy(cur_label.astype(np.uint8)).cuda())
        return ovr_label

    def diganosis(outputs_ovr_soft):
        k = len(outputs_ovr_soft)
        pseudo = [torch.argmax(i[:, 1, ...], dim=1) for i in outputs_ovr_soft]
        pseudo = torch.stack(pseudo, dim=0)
        diag_matrix = torch.sum(pseudo, dim=0)

        return (diag_matrix == 0) | (diag_matrix == (2*k-1))

    def generate_pseudo_label_withmask(outputs_ovr_soft):
        background = [i[:, 0, ...] for i in outputs_ovr_soft]
        background = torch.stack(background, dim=0)
        background = torch.mean(background, dim=0)

        logits_map = [background]
        mask = diganosis(outputs_ovr_soft)
        for i in range(len(outputs_ovr_soft)):
            logit = outputs_ovr_soft[i][:, 1, ...]
            logits_map.append(logit)
        logits_map = torch.stack(logits_map, dim=1)

        mask_with_channel = [mask] * (args.num_classes)
        mask_with_channel = torch.stack(mask_with_channel, dim=1)
        return logits_map.detach(), mask.detach(), mask_with_channel.detach()

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose(
            [OVRWeakStrongAugment(args.patch_size, args.num_classes)]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)

    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    model = create_model()
    ema_model = create_model(ema=True)

    iter_num = 0
    start_epoch = 0

    # instantiate optimizers
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    # if restoring previous models:

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # set to train
    model.train()
    ema_model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    dice_loss_ovr = losses.DiceLoss(3)

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0

    iter_num = int(iter_num)

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)

    for epoch_num in iterator:
        total_loader = zip(trainloader, trainloader)
        for i_batch, (sampled_batch, sample_batch_mix) in enumerate(total_loader):
            weak_batch, strong_batch, label_batch, ovr_label = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
                sampled_batch["ovr_label"]
            )

            weak_batch, strong_batch, label_batch, ovr_label = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
                [item.cuda() for item in ovr_label]
            )
            weak_batch_mix, strong_batch_mix, label_batch_mix = (
                sample_batch_mix["image_weak"],
                sample_batch_mix["image_strong"],
                sample_batch_mix["label_aug"],
            )
            weak_batch_mix, strong_batch_mix, label_batch_mix = (
                weak_batch_mix.cuda(),
                strong_batch_mix.cuda(),
                label_batch_mix.cuda(),
            )

            img_mask = random_mask(
                weak_batch.shape[0], weak_batch.shape[2], args.cut_p)
            img_mask_channel = img_mask.unsqueeze(1)
            with torch.no_grad():
                out_mix, out_mix_ovr = ema_model(weak_batch_mix)
                soft_mix = torch.softmax(out_mix, dim=1)
                pred_mix = torch.argmax(soft_mix[args.labeled_bs:], dim=1)
                mix_conf_mask = (torch.max(soft_mix, dim=1)
                                 [0] > args.conf_thresh)

                soft_mix_ovr = [torch.softmax(i, dim=1) for i in out_mix_ovr]
                pred_mix_ovr = [torch.argmax(i, dim=1) for i in soft_mix_ovr]
                mask_ovr_mix = [torch.max(i, dim=1)[0] > args.conf_thresh for i in soft_mix_ovr]
                
                mix_logits_ovr, mix_conf_mask_ovr, _ = generate_pseudo_label_withmask(
                    soft_mix_ovr)
                mix_pseudo_outputs_ovr = torch.argmax(
                    mix_logits_ovr[args.labeled_bs:].detach(), dim=1, keepdim=False)

            strong_batch[args.labeled_bs:] = img_mask_channel[args.labeled_bs:] * strong_batch[args.labeled_bs:] + (
                1 - img_mask_channel[args.labeled_bs:]) * strong_batch_mix[args.labeled_bs:]

            # outputs for model
            outputs_weak, outputs_weak_ovr = model(weak_batch)
            outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
            outputs_weak_ovr_soft = []
            for i in range(args.num_classes - 1):
                outputs_weak_ovr_soft.append(
                    torch.softmax(outputs_weak_ovr[i], dim=1))

            outputs_strong, outputs_strong_ovr = model(strong_batch)
            outputs_strong_soft = torch.softmax(outputs_strong, dim=1)
            outputs_strong_ovr_soft = [torch.softmax(i, dim=1) for i in outputs_strong_ovr]
            
            consistency_weight = get_current_consistency_weight(
                iter_num // (max_iterations/args.consistency_rampup))

            # supervised loss
            sup_loss = ce_loss(outputs_weak[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + dice_loss(
                outputs_weak_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )
            sup_loss_ovr = 0
            for i in range(args.num_classes - 1):
                sup_loss_ovr += ce_loss(outputs_weak_ovr[i][: args.labeled_bs], ovr_label[i][: args.labeled_bs].long(),) + dice_loss_ovr(
                    outputs_weak_ovr_soft[i][: args.labeled_bs],
                    ovr_label[i][: args.labeled_bs].unsqueeze(1),
                )
            sup_loss += sup_loss_ovr/(args.num_classes - 1)
            with torch.no_grad():
                ema_output, ema_ovr_output = ema_model(weak_batch)
                ema_outputs_soft = torch.softmax(ema_output, dim=1)
                ema_ovr_output_soft = [torch.softmax(i, dim=1) for i in ema_ovr_output]

                conf_mask = (torch.max(ema_outputs_soft, dim=1)[
                    0] > args.conf_thresh)
                conf_mask = img_mask*conf_mask + (1-img_mask)*mix_conf_mask
                mask_channel = torch.stack([conf_mask] * args.num_classes, dim=1)

                pseudo_outputs = torch.argmax(
                    ema_outputs_soft[args.labeled_bs:].detach(), dim=1, keepdim=False)
                pseudo_outputs = img_mask[args.labeled_bs:] * \
                    pseudo_outputs + (1-img_mask[args.labeled_bs:])*pred_mix
                pseudo_lab4ovr = generate_ovr_pseudu_label(
                    pseudo_outputs, args)
                mask_ovr_tmp = [torch.max(i, dim=1)[0] > args.conf_thresh for i in ema_ovr_output_soft]
                mask_ovr = [img_mask*i + (1-img_mask)*j for i, j in zip(mask_ovr_tmp, mask_ovr_mix)]
                mask_ovr_cnl = [torch.stack([i]*3, dim=1) for i in mask_ovr]


                pseudo_ovr_tmp = []
                for i in range(args.num_classes - 1):
                    pseudo_ovr_tmp.append(
                        torch.argmax(ema_ovr_output_soft[i], dim=1))
                pseudo_ovr = []
                for i, j in zip(pseudo_ovr_tmp, pred_mix_ovr):
                    pseudo_ovr.append(
                        img_mask * i + (1-img_mask)*j)

                logits_ovr, conf_mask_ovr, mask_channel_ovr = generate_pseudo_label_withmask(
                    ema_ovr_output_soft)
                conf_mask_ovr = img_mask*conf_mask_ovr + \
                    (1-img_mask)*mix_conf_mask_ovr
                mask_channel_ovr = torch.stack(
                    [conf_mask_ovr]*args.num_classes, dim=1)
                pseudo_outputs_ovr = torch.argmax(
                    logits_ovr[args.labeled_bs:].detach(), dim=1, keepdim=False)
                pseudo_outputs_ovr = img_mask[args.labeled_bs:] * pseudo_outputs_ovr + (
                    1-img_mask[args.labeled_bs:])*mix_pseudo_outputs_ovr

                # ensemble pseudo_outputs_ovr and pseudo_outputs
                ensemble_mask = (pseudo_outputs == pseudo_outputs_ovr)
                ensemble_mask_channel = [ensemble_mask] * args.num_classes
                ensemble_mask_channel = torch.stack(
                    ensemble_mask_channel, dim=1)
                ensemble_pseudo = pseudo_outputs * ensemble_mask
                ensemble_pseudo4ovr = generate_ovr_pseudu_label(
                    ensemble_pseudo, args)

            # unsupervised loss
            unsup_loss = (
                cross_entropy_masked(
                    outputs_strong[args.labeled_bs:], pseudo_outputs.long(), conf_mask[args.labeled_bs:])
                + dice_loss(
                    outputs_strong_soft[args.labeled_bs:],
                    pseudo_outputs.unsqueeze(1),
                    mask=mask_channel[args.labeled_bs:]
                )
            )

            unsup_loss += (
                cross_entropy_masked(
                    outputs_strong[args.labeled_bs:], pseudo_outputs_ovr.long(), conf_mask_ovr[args.labeled_bs:])
                + dice_loss(
                    outputs_strong_soft[args.labeled_bs:],
                    pseudo_outputs_ovr.unsqueeze(1),
                    mask=mask_channel_ovr[args.labeled_bs:]
                )
            )

            unsup_loss += (
                cross_entropy_masked(
                    outputs_strong[args.labeled_bs:], ensemble_pseudo.long(), ensemble_mask)
                + dice_loss(
                    outputs_strong_soft[args.labeled_bs:],
                    ensemble_pseudo.unsqueeze(1),
                    mask=ensemble_mask_channel
                )
            )

            unsup_loss_ovr = 0
            for i in range(args.num_classes - 1):
                unsup_loss_ovr += (
                    cross_entropy_masked(
                        outputs_strong_ovr[i][args.labeled_bs:], pseudo_lab4ovr[i].long(), conf_mask[args.labeled_bs:])
                    + dice_loss_ovr(
                        outputs_strong_ovr_soft[i][args.labeled_bs:],
                        pseudo_lab4ovr[i].unsqueeze(1),
                        mask=mask_channel[args.labeled_bs:, 0:3]
                    )
                )
                unsup_loss_ovr += (
                    cross_entropy_masked(
                        outputs_strong_ovr[i][args.labeled_bs:], pseudo_ovr[i][args.labeled_bs:].long(), conf_mask_ovr[args.labeled_bs:])
                    + dice_loss_ovr(
                        outputs_strong_ovr_soft[i][args.labeled_bs:],
                        pseudo_ovr[i][args.labeled_bs:].unsqueeze(1),
                        mask=mask_channel_ovr[args.labeled_bs:, 0:3]
                    )
                )
                unsup_loss_ovr += (
                    cross_entropy_masked(
                        outputs_strong_ovr[i][args.labeled_bs:], ensemble_pseudo4ovr[i].long(), ensemble_mask)
                    + dice_loss_ovr(
                        outputs_strong_ovr_soft[i][args.labeled_bs:],
                        ensemble_pseudo4ovr[i].unsqueeze(1),
                        mask=ensemble_mask_channel[:, 0:3]
                    )
                )
            unsup_loss += unsup_loss_ovr/(args.num_classes - 1)

            loss = sup_loss + consistency_weight * unsup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update ema model
            update_ema_variables(
                model, ema_model, args.ema_decay, iter_num, buffer=False)

            # update learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num = iter_num + 1

            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar(
                "consistency_weight/consistency_weight", consistency_weight, iter_num)
            writer.add_scalar("loss/model_loss", loss, iter_num)
            writer.add_scalar("loss/sup_loss_ovr", sup_loss_ovr, iter_num)
            # writer.add_scalar("loss/unsup_loss_ovr", unsup_loss_ovr, iter_num)
            logging.info("iteration %d : model loss : %f ovr_sup: %f" % (
                iter_num, loss.item(), (sup_loss_ovr/(args.num_classes - 1)).item()))
            if iter_num % 50 == 0:
                # show weakly augmented image
                image = weak_batch[1, 0:1, :, :]
                writer.add_image("train/Image", image, iter_num)
                # show strongly augmented image
                image_strong = strong_batch[1, 0:1, :, :]
                writer.add_image("train/StrongImage", image_strong, iter_num)
                # show model prediction (strong augment)
                outputs_strong = torch.argmax(
                    outputs_strong_soft, dim=1, keepdim=True)
                writer.add_image("train/model_Prediction",
                                 outputs_strong[1, ...] * 50, iter_num)
                # show ground truth label
                labs = label_batch[1, ...].long().unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)
                # show generated pseudo label
                pseudo_labs = torch.argmax(
                    outputs_weak_soft.detach(), dim=1, keepdim=False)[1, ...].unsqueeze(0) * 50
                writer.add_image("train/PseudoLabel", pseudo_labs, iter_num)

                pseudo_labs_ovr = torch.argmax(
                    logits_ovr.detach(), dim=1, keepdim=False)[1, ...].unsqueeze(0) * 50
                writer.add_image("train/PseudoLabel_ovr",
                                 pseudo_labs_ovr, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                ema_model.eval()

                metric_list = 0.0
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(valloader):
                        metric_i = test_single_volume_ovr(
                            sampled_batch["image"],
                            sampled_batch["label"],
                            ema_model,
                            classes=num_classes,
                        )
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        "info/model_val_{}_dice".format(class_i + 1),
                        metric_list[class_i, 0],
                        iter_num,
                    )

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar("info/model_val_mean_dice",
                                  performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path,
                        "model_iter_{}_dice_{}.pth".format(
                            iter_num, round(best_performance, 4)),
                    )
                    save_best = os.path.join(
                        snapshot_path, "{}_best_model.pth".format(args.model))
                    # util.save_checkpoint(
                    #     epoch_num, model, optimizer, loss, save_mode_path)
                    util.save_checkpoint(
                        epoch_num, ema_model, optimizer, loss, save_best)

                logging.info(
                    "iteration %d : model_mean_dice : %f " % (
                        iter_num, performance)
                )
                model.train()
                ema_model.train()
                

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "model_iter_" + str(iter_num) + ".pth")
                util.save_checkpoint(
                    epoch_num, model, optimizer, loss, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

if __name__ == "__main__":
    torch.set_num_threads(4)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
    