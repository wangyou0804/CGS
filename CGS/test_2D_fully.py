import argparse
import os
import shutil

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
import torch.nn as nn
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--save_model_path', type=str,
                    default=None, help='save_model_path')



def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd95, asd
    else:
        return 0, 0, 50, 10.


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == 'OVRUNet':
                out_main, _ = net(input)
            else:
                out_main = net(input)

            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    metric = []

    for i in range(1, FLAGS.num_classes):
        metric.append(calculate_metric_percase(prediction == i, label == i))


    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "model/{}_{}/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "model/{}_{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_model_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    if FLAGS.save_model_path:
        save_model_path = FLAGS.save_model_path
    model = torch.load(save_model_path)
    if "state_dict" in model.keys():
        model = model["state_dict"]

    net.load_state_dict(model)

    print("init weight from {}".format(save_model_path))
    net.eval()

    total = [[0, 0, 0, 0]] * (FLAGS.num_classes - 1)

    for case in tqdm(image_list):
        metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        for i in range(FLAGS.num_classes - 1):
            total[i] = [i + j for i, j in zip(total[i], metric[i])]
    avg_metric = []
    for i in total:
        tmp = []
        for j in i:
            tmp.append(j / len(image_list))
        avg_metric.append(tmp)
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print(np.mean(metric, axis=0))
