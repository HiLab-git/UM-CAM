"""
Generate noisy 2D segmentation labels by random dilation, erosion, shift or edge distortion. 
"""
import os
from PIL import Image
from scipy import ndimage
import numpy as np
import numpy as np
import random
import SimpleITK as sitk
from medpy import metric

def create_circle_mask_on_edge(label, r_range, sample_ratio):
    H, W = label.shape
    edge = ndimage.binary_dilation(label) - label
    y, x = np.where(edge > 0)
    edge_length = len(y)
    idx  = random.sample(range(edge_length), int(edge_length * sample_ratio))
    ys, xs = y[idx], x[idx]
    
    # create mask with circle
    mask = np.zeros_like(label)
    num  = len(xs)
    for i in range(num):
        yi, xi = ys[i], xs[i]
        r = random.randint(r_range[0], r_range[1])
        for h in range(-r, r):
            for w in range(-r, r):
                yt, xt = yi + h, xi + w 
                if(yt < 0 or yt >= H or xt <  0 or xt >= W):
                    continue
                if((xt - xi)* (xt - xi) + (yt - yi)* (yt - yi) < r*r):
                    mask[yt, xt] = 1
    return mask

def random_edge_distort(label, r_range, sample_ratio):
    mask1 = create_circle_mask_on_edge(label, r_range, sample_ratio)
    out   = np.maximum(mask1, label)
    mask2 = create_circle_mask_on_edge(out, r_range, sample_ratio)
    out = (1 - mask2) * out
    return out


def add_random_nosie_to_label(label, r_range, s_range):
    p = random.random()
    if(p < 0.25):
        r = random.randint(r_range[0], r_range[1])
        label = ndimage.binary_dilation(label, iterations = r)
    elif(p < 0.50):
        r = random.randint(r_range[0], r_range[1])
        label = ndimage.binary_erosion(label, iterations = r)
    elif(p < 0.75):
        s_x = random.randint(s_range[0], s_range[1])
        s_y = random.randint(s_range[0], s_range[1])
        label = random_shift(label, (s_x, s_y))
    else:
        label = random_edge_distort(label, r_range, sample_ratio = 0.1)
        
    return label 


def random_shift(label, shift):
    label_shift = np.zeros_like(label)
    label_shift[shift[0]:, shift[1]:] = label[:-shift[0], :-shift[1]]
    return label_shift


def generate_noise(input_name, output_name, radius_range=None, shift_range=None, radius_ratio=None, shift_ratio=None):
    lab_itk = sitk.ReadImage(input_name)
    lab = sitk.GetArrayFromImage(lab_itk).squeeze()
    lab_max = lab.max()
    lab = np.asarray(lab > 0, np.uint8)

    H, W = lab.shape
    if radius_range is None:
        radius_range = (int(radius_ratio[0] * min(H, W)), int(radius_ratio[1] * min(H, W)))

    if shift_range is None:
        shift_range_min = max(1, int(shift_ratio[0] * min(H, W)))
        shift_range_max = int(shift_ratio[1] * min(H, W))
        shift_range = (shift_range_min, shift_range_max)
        
    lab_noise = add_random_nosie_to_label(lab, radius_range, shift_range) * lab_max
    lab_noise_itk = sitk.GetImageFromArray(np.expand_dims(lab_noise, 0))
    lab_noise_itk.CopyInformation(lab_itk)
    sitk.WriteImage(lab_noise_itk, output_name)
    

if __name__ == "__main__":
    random.seed(2024)
    np.random.seed(2024)
    
    noise_range = 50
    radius_range, shift_range = (1, noise_range), (1, noise_range)

    dir_root_slice = 'slice/label/train'
    dir_root_volume = 'volume/label/train'

    dir_pseudo_slice = f'slice/label_noise_{noise_range}/train'
    os.makedirs(dir_pseudo_slice, exist_ok=True)

    volume_names = sorted(os.listdir(dir_root_volume))
    volume_names = [name.split('.')[0] for name in volume_names]

    np.random.shuffle(volume_names)
    
    cases = sorted(os.listdir(dir_root_slice))
    for case in cases:
        generate_noise(f'{dir_root_slice}/{case}', f'{dir_pseudo_slice}/{case}', radius_range, shift_range)
        


    """assess the quality of pseudo label"""
    volume_names = sorted(os.listdir(dir_root_volume))
    volume_names = [name.split('.')[0] for name in volume_names]

    dice_all, hd95_all = [], []
    for name in volume_names:
        label = sitk.GetArrayFromImage(sitk.ReadImage(f'{dir_root_volume}/{name}.nii.gz')).squeeze()
        D = label.shape[0]

        pseudo_label = np.zeros_like(label)
        for i in range(D):
            pseudo_label[i] = sitk.GetArrayFromImage(sitk.ReadImage(f'{dir_pseudo_slice}/{name}_slice_{i+1}.nii.gz')).squeeze()

        dice = metric.dc(pseudo_label, label)
        dice_all.append(dice)

        hd95 = metric.binary.hd95(pseudo_label, label)
        hd95_all.append(hd95)
        print(name, dice, hd95)

    dice_all = np.array(dice_all)
    hd95_all = np.array(hd95_all)

    print('dice:', dice_all.mean() * 100.0, dice_all.std() * 100.0)
    print('hd95:', hd95_all.mean(), hd95_all.std())