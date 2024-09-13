import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn.functional as F

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, asd


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            gt = label[ind, :, :]
            if gt.sum() != 0:
                slice = image[ind, :, :]
                x, y = slice.shape[0], slice.shape[1]
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
                net.eval()
                with torch.no_grad():
                    out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
                    out = out.cpu().detach().numpy()
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[ind] = pred
    else:
        prediction = np.zeros_like(label)
        if label.sum() != 0:
            input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
                prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
