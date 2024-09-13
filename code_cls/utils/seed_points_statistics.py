import os
import cv2
import numpy as np
import SimpleITK as sitk
import argparse
from skimage import measure
from skimage.measure import label
from PIL import Image
import torch
import FastGeodis
from skimage.measure import regionprops

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def largestConnectComponent(binaryimg):
    label_image, num = measure.label(binaryimg, background=0, return_num=True)
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    
    if len(areas) > 1:
        for region in measure.regionprops(label_image):
            if region.area < areas[-1]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    label_image = label_image.astype(np.int8)
    label_image[np.where(label_image > 0)] = 1
    return label_image


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_seed_points(bw_img, margin_x=5, margin_y=5):
    labeled_img, num = label(bw_img, background=0, return_num=True)

    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i

    lcc = np.zeros(bw_img.shape)
    lcc[np.where(labeled_img == max_label)] = 1
    
    """calculate the centroid of the largest connect component"""
    bw_img = np.uint8(bw_img)    
    contours, hierarchy = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    """select the centroid, and the rectangular vertexes"""
    M = cv2.moments(contours[0])
    cx = int(M['m10'] / max(M['m00'], 1e-8))
    cy = int(M['m01'] / max(M['m00'], 1e-8))
    (x, y, w, h) = cv2.boundingRect(contours[0])
    
    return lcc, (cy, cx), (y-margin_y, x-margin_x, h+margin_y, w+margin_x)


def get_seed_points_3d(bw_img):
    assert bw_img.sum() > 0
    
    labeled_img, _ = label(bw_img, background=0, return_num=True)
    max_area = 0
    max_region = None
    for region in regionprops(labeled_img):
        if region.area > max_area:
            max_area = region.area
            max_region = region

    if max_region is not None:
        centroid = max_region.centroid
        bbox = max_region.bbox
        
    return centroid, bbox


def get_seed_points_minarea(bw_img, margin_x=5, margin_y=5):
    labeled_img, num = label(bw_img, background=0, return_num=True)
    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i

    lcc = np.zeros(bw_img.shape)
    lcc[np.where(labeled_img == max_label)] = 1

    """calculate the centroid of the largest connect component"""
    bw_img = np.uint8(bw_img)
    contours, hierarchy = cv2.findContours(bw_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    """select the centroid, and the rectangular vertexes"""
    M = cv2.moments(contours[0])
    cx = int(M['m10'] / max(M['m00'], 1e-8))
    cy = int(M['m01'] / max(M['m00'], 1e-8)) 
    rect = cv2.minAreaRect(contours[0])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    print(points.dtype)
    print(points[0].shape())
    return lcc, (cy, cx), points


def seeds_map(img, seed_points, margin=5):
    H, W = img.shape
    S = np.ones((H, W), np.uint8)
    for i in range(len(seed_points)):
        if seed_points[i][0] == H:
            if seed_points[i][1] == W:
                S[H-margin, W-margin] = 0
            else:
                S[H-margin, seed_points[i][1]] = 0
        else:
            if seed_points[i][1] == W:
                S[seed_points[i][0], W-margin] = 0
            else:
                S[seed_points[i][0], seed_points[i][1]] = 0
    return S


def seeds_map_bbox(img, seed_points, margin=5):
    H, W = img.shape
    S = np.ones((H, W), np.uint8)
    if len(seed_points) == 1:
        S[seed_points[0][0], seed_points[0][1]] = 0
    else:
        S[seed_points[0][0]: seed_points[1][0], seed_points[0][1]] = 0
        S[seed_points[2][0]: seed_points[3][0] - 1, seed_points[2][1] - 1] = 0
        S[seed_points[0][0], seed_points[0][1]: seed_points[2][1] - 1] = 0
        S[seed_points[1][0] - 1, seed_points[1][1]: seed_points[3][1] - 1] = 0
    return S


def max_min_norm(input):
    return (input - input.min()) / (input.max() - input.min())


def plot_seeds(mask, seed_points_fg, seed_points_bg, marker_size=3):
    mask = np.asarray(mask, np.float32)
    mask = mask * 255.0
    Out = np.array([mask, mask, mask])
    Out = np.transpose(Out, (1,2,0))
    Out = Image.fromarray(np.uint8(Out))
    
    if seed_points_fg == None:
        print('empty foreground seed points')
    else:
        for idx in range(len(seed_points_fg)):
            x, y = seed_points_fg[idx]
            for i in range(x-marker_size, x+marker_size):
                for j in range(y-marker_size, y+marker_size):
                    Out.putpixel((j, i), (255, 0, 0))
        
    if seed_points_bg == None:
        print('empty background seed points')
    else:
        for idx in range(len(seed_points_bg)):
            x, y = seed_points_bg[idx]
            for i in range(x-marker_size, x+marker_size):
                for j in range(y-marker_size, y+marker_size):
                    Out.putpixel((j, i), (0, 255, 0))

    return Out


def geodis_fast(image, seed_points, v=1e10, lamb=1.0, iterations=1):
    mask_pt = seeds_map(image,seed_points)
    image_pt = torch.from_numpy(image).unsqueeze_(0).unsqueeze_(0)
    mask_pt = torch.from_numpy(mask_pt).unsqueeze_(0).unsqueeze_(0)
    
    geodesic_dist = FastGeodis.generalised_geodesic2d(image_pt, mask_pt, v, lamb, iterations)
    geodesic_dist = np.squeeze(geodesic_dist.cpu().numpy())
    
    return geodesic_dist


def seeds_map_3d(img, seed_points, margin=[5, 5, 5]):
    # N, H, W = img.shape
    S = np.ones_like(img, np.uint8)
    for i in range(len(seed_points)):
        S[seed_points[i][0] -margin[0], seed_points[i][1] - margin[1], seed_points[i][2] - margin[2]] = 0
    return S


def geodis_fast_3d(image, seed_points, v=1e10, lamb=1.0, iterations=1, spacing=[1,1,1], margin=[5, 5, 5]):
    mask_pt = seeds_map_3d(image,seed_points, margin)
    image_pt = torch.from_numpy(image).unsqueeze_(0).unsqueeze_(0)
    mask_pt = torch.from_numpy(mask_pt).unsqueeze_(0).unsqueeze_(0)
    
    geodesic_dist = FastGeodis.generalised_geodesic3d(image_pt, mask_pt, spacing, v, lamb, iterations)
    geodesic_dist = np.squeeze(geodesic_dist.cpu().numpy())
    return geodesic_dist
