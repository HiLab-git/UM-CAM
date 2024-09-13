import os
from PIL import Image
import numpy as np
from skimage import measure


def map_scalar_to_color(x):
    x_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    c_list = [[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
    for i in range(len(x_list)):
        if x <= x_list[i + 1]:
            x0 = x_list[i]
            x1 = x_list[i + 1]
            c0 = c_list[i]
            c1 = c_list[i + 1]
            alpha = (x - x0) / (x1 - x0)
            c = [c0[j] * (1 - alpha) + c1[j] * alpha for j in range(3)]
            c = [int(item) for item in c]
            return tuple(c)


def get_fused_heat_map(image, att):
    [H, W] = image.size
    img = Image.new("RGB", image.size, (255, 0, 0))

    for i in range(H):
        for j in range(W):
            p0 = image.getpixel((i, j))
            alpha = att.getpixel((i, j))
            p1 = map_scalar_to_color(alpha)
            alpha = 0.5 + alpha * 0.3
            p = [int(p0 * (1 - alpha) + p1[c] * alpha) for c in range(3)]
            p = tuple(p)
            img.putpixel((i, j), p)
    return img


def largestConnectComponent(img):
    binaryimg = img
    label_image, num = measure.label(binaryimg, background=0, return_num=True)
    areas = [r.area for r in measure.regionprops(label_image)]
    areas.sort()
    # print(num)
    if len(areas) > 1:
        for region in measure.regionprops(label_image):
            if region.area < areas[-1] / 2:
                # print(region.area)
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    label_image = label_image.astype(np.int8)
    label_image[np.where(label_image > 0)] = 1
    return label_image
