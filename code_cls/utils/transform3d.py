# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import json
import math
import random
import numpy as np
import os
from scipy import ndimage
from code_cls.utils.image_process import *
import matplotlib.pyplot as plt
from PIL import Image


__all__ = ["Compose", "Normalize", "Rescale", "RandomFlip", "RandomRotate", "Pad", 
           "CropWithBoundingBox", "RandomCrop", "CenterCrop", "ChannelWiseGammaCorrection",
           "ChannelWiseNormalize", "ChannelWiseThreshold", "ChannelWiseThresholdWithNormalize",
           "ReduceLabelDim", "LabelConvert", "LabelConvertNonzero", "LabelToProbability",
           "ProbabilityToDistance", "RegionSwop"]

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
class Normalize(object):
    """Nomralize the image (shape [C, D, H, W] or [C, H, W]) for each channel

    Args:
        mean (None or tuple/list): The mean values along each channel.
        std  (None or tuple/list): The std values along each channel.
            if mean and std are None, calculate them from non-zero region
        chns (None, or tuple/list): The list of channel indices
        zero_to_random (bool, or tuple/list or bool): indicate whether zero values
             in each channel is replaced  with random values.
    """
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.chns = len(min)

    def __call__(self, sample):
        image = sample['image']
        B, C, H, W = image.shape
        new_image = np.zeros([B, self.chns, H, W], dtype='float32')
        for chn in range(self.chns):
            chn_min = self.min[chn]
            chn_max = self.max[chn]
            chn_norm = (image - chn_min) / (chn_max - chn_min)
            chn_norm[chn_norm < 0] = 0
            chn_norm[chn_norm > 1] = 1
            chn_norm = chn_norm * 255.0
            new_image[:, chn, :, :] = chn_norm
        sample['image'] = new_image

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple/list or int): Desired output size. 
            If tuple/list, output_size should in the format of [D, H, W] or [H, W].
            Channel number is kept the same as the input. If D is None, the input image
            is only reslcaled in 2D.
            If int, the smallest axis is matched to output_size keeping 
            aspect ratio the same.
    """

    def __init__(self, output_size_img, output_size_GT=None, inverse=True):
        assert isinstance(output_size_img, (int, list, tuple))
        if output_size_GT is not None:
            assert isinstance(output_size_GT, (int, list, tuple))
            self.output_size_GT = output_size_GT

        self.output_size_img = output_size_img
        self.inverse = inverse

    def scale_image(self, image):
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        
        if isinstance(self.output_size_img, (list, tuple)):
            output_size = self.output_size_img
            if (output_size[0] is None):
                output_size[0] = input_shape[1]
            assert (len(output_size) == input_dim)
        else:
            min_edge = min(input_shape[1:])
            output_size = [self.output_size_img * input_shape[i + 1] / min_edge for i in range(input_dim)]
        scale = [(output_size[i] + 0.0) / input_shape[1:][i] for i in range(input_dim)]
        scale = [1.0] + scale
        image = ndimage.interpolation.zoom(image, scale, order=1)

        return image, scale

    def scale_GT(self, image):
        input_shape = image.shape
        input_dim = len(input_shape) - 1

        if isinstance(self.output_size_GT, (list, tuple)):
            output_size = self.output_size_GT
            if (output_size[0] is None):
                output_size[0] = input_shape[1]
            assert (len(output_size) == input_dim)
        else:
            min_edge = min(input_shape[1:])
            output_size = [self.output_size_GT * input_shape[i + 1] / min_edge for i in range(input_dim)]
        scale = [(output_size[i] + 0.0) / input_shape[1:][i] for i in range(input_dim)]
        scale = [1.0] + scale
        image = ndimage.interpolation.zoom(image, scale, order=1)

        return image

    def __call__(self, sample):
        image = sample['image']
        image, scale = self.scale_image(image)
        input_shape = image.shape
        sample['image'] = image
        sample['Rescale_origin_shape'] = json.dumps(input_shape)

        # if ('label' in sample):
        #     label = sample['label']
        #     label = ndimage.interpolation.zoom(label, scale, order=0)
        #     sample['label'] = label

        if ('gt_map' in sample):
            gt_img = sample['gt_map']
            gt_img = self.scale_GT(gt_img).squeeze()
            sample['gt_map'] = gt_img

        # if ('att_map' in sample):
        #     gt_img = sample['att_map']
        #     gt_img = self.scale_GT(gt_img).squeeze()
        #     sample['att_map'] = gt_img

        """save the image after scaling"""
        # save_name = "E:/Python/PHD/Fetal/CAM/DRS-main/scripts/result/localization_maps/100epoch_1202_scale/nii_origin/" + os.path.basename(sample['names'])
        # save_array_as_nifty_volume(sample['image'][0], save_name)
        # save_name_png = "E:/Python/PHD/Fetal/CAM/DRS-main/scripts/result/localization_maps/100epoch_1202_scale/png_origin/" + os.path.basename(sample['names']).split('.')[0] + '.png'
        # image_png = np.uint8(sample['image'].squeeze() * 255)
        # image_png = Image.fromarray(image_png)
        # image_png.save(save_name_png)

        return sample

    def inverse_transform_for_prediction(self, sample):
        ''' rescale sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        if(isinstance(sample['Rescale_origin_shape'], list) or \
            isinstance(sample['Rescale_origin_shape'], tuple)):
            origin_shape = json.loads(sample['Rescale_origin_shape'][0])
        else:
            origin_shape = json.loads(sample['Rescale_origin_shape'])
        origin_dim   = len(origin_shape) - 1
        predict = sample['predict']
        input_shape = predict.shape
        scale = [(origin_shape[1:][i] + 0.0)/input_shape[2:][i] for \
                i in range(origin_dim)]
        scale = [1.0, 1.0] + scale

        output_predict = ndimage.interpolation.zoom(predict, scale, order = 1)
        sample['predict'] = output_predict
        return sample

class RandomFlip(object):
    """
    random flip the image (shape [C, D, H, W] or [C, H, W]) 
    Args:
        flip_depth (bool) : random flip along depth axis or not, only used for 3D images
        flip_height (bool): random flip along height axis or not
        flip_width (bool) : random flip along width axis or not
    """
    def __init__(self, flip_depth, flip_height, flip_width, inverse):
        self.flip_depth  = flip_depth
        self.flip_height = flip_height
        self.flip_width  = flip_width
        self.inverse = inverse

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        flip_axis = []
        if(self.flip_width):
            if(random.random() > 0.5):
                flip_axis.append(-1)
        if(self.flip_height):
            if(random.random() > 0.5):
                flip_axis.append(-2)
        if(input_dim == 3 and self.flip_depth):
            if(random.random() > 0.5):
                flip_axis.append(-3)
        if(len(flip_axis) > 0):
            # use .copy() to avoid negative strides of numpy array
            # current pytorch does not support negative strides
            sample['image'] = np.flip(image, flip_axis).copy()

        sample['RandomFlip_Param'] = json.dumps(flip_axis)
        # if('label' in sample and len(flip_axis) > 0):
        #     sample['label'] = np.flip(sample['label'] , flip_axis).copy()
        # if ('image' in sample):
        #     sample['image'] = np.flip(sample['image'], flip_axis).copy()
        if ('gt_map' in sample and len(flip_axis) > 0):
            sample['gt_map'] = np.flip(sample['gt_map'], flip_axis).copy()
        if('contour' in sample and len(flip_axis) > 0):
            sample['contour'] = np.flip(sample['contour'] , flip_axis).copy()
        
        return sample

    def  inverse_transform_for_prediction(self, sample):
        ''' flip sample['predict'] (5D or 4D) to the original direction.
         assume batch size is 1, otherwise flip parameter may be different for 
         different elemenets in the batch.

        flip_axis is a list as saved in __call__().'''
        if(isinstance(sample['RandomFlip_Param'], list) or \
            isinstance(sample['RandomFlip_Param'], tuple)):
            flip_axis = json.loads(sample['RandomFlip_Param'][0]) 
        else:
            flip_axis = json.loads(sample['RandomFlip_Param']) 
        if(len(flip_axis) > 0):
            sample['predict']  = np.flip(sample['predict'] , flip_axis)
        return sample

class RandomRotate(object):
    """
    random rotate the image (shape [C, D, H, W] or [C, H, W]) 
    Args:
        angle_range_d (tuple/list/None) : rorate angle range along depth axis (degree),
               only used for 3D images
        angle_range_h (tuple/list/None) : rorate angle range along height axis (degree)
        angle_range_w (tuple/list/None) : rorate angle range along width axis (degree)
    """
    def __init__(self, angle_range_d, angle_range_h, angle_range_w, inverse):
        self.angle_range_d  = angle_range_d
        self.angle_range_h  = angle_range_h
        self.angle_range_w  = angle_range_w
        self.inverse = inverse

    def __apply_transformation(self, image, transform_param_list, order = 1):
        """
        apply rotation transformation to an ND image
        Args:
            image (nd array): the input nd image
            transform_param_list (list): a list of roration angle and axes
            order (int): interpolation order
        """
        for angle, axes in transform_param_list:
            image = ndimage.rotate(image, angle, axes, reshape = False, order = order)
        return image

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        
        transform_param_list = []
        if(self.angle_range_d is not None):
            angle_d = np.random.uniform(self.angle_range_d[0], self.angle_range_d[1])
            transform_param_list.append([angle_d, (-1, -2)])
        if(input_dim == 3):
            if(self.angle_range_h is not None):
                angle_h = np.random.uniform(self.angle_range_h[0], self.angle_range_h[1])
                transform_param_list.append([angle_h, (-1, -3)])
            if(self.angle_range_w is not None):
                angle_w = np.random.uniform(self.angle_range_w[0], self.angle_range_w[1])
                transform_param_list.append([angle_w, (-2, -3)])
        assert(len(transform_param_list) > 0)

        sample['image'] = self.__apply_transformation(image, transform_param_list, 1)
        sample['RandomRotate_Param'] = json.dumps(transform_param_list)
        # if('label' in sample):
        #     sample['label'] = self.__apply_transformation(sample['label'], transform_param_list, 0)
        if ('gt_map' in sample):
            sample['gt_map'] = self.__apply_transformation(sample['gt_map'], transform_param_list, 0)
        if('contour' in sample):
            sample['contour'] = self.__apply_transformation(sample['contour'], transform_param_list, 0)
        return sample

    def  inverse_transform_for_prediction(self, sample):
        ''' rorate sample['predict'] (5D or 4D) to the original direction.
        assume batch size is 1, otherwise rotate parameter may be different for 
        different elemenets in the batch.

        transform_param_list is a list as saved in __call__().'''
        # get the paramters for invers transformation
        if(isinstance(sample['RandomRotate_Param'], list) or \
            isinstance(sample['RandomRotate_Param'], tuple)):
            transform_param_list = json.loads(sample['RandomRotate_Param'][0]) 
        else:
            transform_param_list = json.loads(sample['RandomRotate_Param']) 
        transform_param_list.reverse()
        for i in range(len(transform_param_list)):
            transform_param_list[i][0] = - transform_param_list[i][0]
        sample['predict'] = self.__apply_transformation(sample['predict'] , 
                                transform_param_list, 1)
        return sample

class Pad(object):
    """
    Pad the image (shape [C, D, H, W] or [C, H, W]) to an new spatial shape, 
    the real output size will be max(image_size, output_size)
    Args:
       output_size (tuple/list): the size along each spatial axis. 
       
    """
    def __init__(self, output_size, ceil_mode = False, inverse = True):
        self.output_size = output_size
        self.ceil_mode   = ceil_mode
        self.inverse = inverse


    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        assert(len(self.output_size) == input_dim)
        if(self.ceil_mode):
            multiple = [int(math.ceil(float(input_shape[1+i])/self.output_size[i]))\
                for i in range(input_dim)]  # when you do not want to pad along D axis, set ceil_mode = True and output_size = [1, H, W]
            output_size = [multiple[i] * self.output_size[i] \
                for i in range(input_dim)]
        else:
            output_size = self.output_size
        margin = [max(0, output_size[i] - input_shape[1+i]) \
            for i in range(input_dim)]

        margin_lower = [int(margin[i] / 2) for i in range(input_dim)]
        margin_upper = [margin[i] - margin_lower[i] for i in range(input_dim)]
        pad = [(margin_lower[i], margin_upper[i]) for  i in range(input_dim)]
        pad = tuple([(0, 0)] + pad)
        if(max(margin) > 0):
            image = np.pad(image, pad, 'reflect')

        sample['image'] = image
        sample['Pad_Param'] = json.dumps((margin_lower, margin_upper))
        if('label' in sample):
            label = sample['label']
            if(max(margin) > 0):
                label = np.pad(label, pad, 'reflect')
            sample['label'] = label
        if('contour' in sample):
            contour = sample['contour']
            if(max(margin) > 0):
                contour = np.pad(contour, pad, 'reflect')
            sample['contour'] = contour
        
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        ''' crop sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        # raise ValueError("not implemented")
        if(isinstance(sample['Pad_Param'], list) or isinstance(sample['Pad_Param'], tuple)):
            params = json.loads(sample['Pad_Param'][0]) 
        else:
            params = json.loads(sample['Pad_Param']) 
        margin_lower = params[0]
        margin_upper = params[1]
        predict = sample['predict']
        if(isinstance(predict, tuple) or isinstance(predict, list)):
            output_predict = []
            for predict_i in predict:
                predict_shape = predict_i.shape
                crop_min = [0, 0] + margin_lower
                crop_max = [predict_shape[2:][i] - margin_upper[i] \
                    for i in range(len(margin_lower))]
                crop_max = list(predict_shape[:2]) + crop_max
                crop_predict = crop_ND_volume_with_bounding_box(predict_i, crop_min, crop_max)
                output_predict.append(crop_predict)
        else:
            predict_shape = predict.shape
            crop_min = [0, 0] + margin_lower
            crop_max = [predict_shape[2:][i] - margin_upper[i] \
                for i in range(len(margin_lower))]
            crop_max = list(predict_shape[:2]) + crop_max

            output_predict = crop_ND_volume_with_bounding_box(predict, crop_min, crop_max)
        sample['predict'] = output_predict
        return sample

class CropWithBoundingBox(object):
    """Crop the image (shape [C, D, H, W] or [C, H, W]) based on bounding box

    Args:
        start (None or tuple/list): The start index along each spatial axis.
            if None, calculate the start index automatically so that 
            the cropped region is centered at the non-zero region.
        output_size (None or tuple/list): Desired spatial output size.
            if None, set it as the size of bounding box of non-zero region 
    """

    def __init__(self, start, output_size, inverse = True):
        self.start = start
        self.output_size = output_size
        self.inverse = inverse


        
    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        bb_min, bb_max = get_ND_bounding_box(image)
        bb_min, bb_max = bb_min[1:], bb_max[1:]

        if(self.start is None):
            if(self.output_size is None):
                crop_min = bb_min, crop_max = bb_max
            else:
                assert(len(self.output_size) == input_dim)
                crop_min = [int((bb_min[i] + bb_max[i] + 1)/2) - int(self.output_size[i]/2) \
                    for i in range(input_dim)]
                crop_min = [max(0, crop_min[i]) for i in range(input_dim)]
                crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        else:
            assert(len(self.start) == input_dim)
            crop_min = self.start
            if(self.output_size is None):
                assert(len(self.output_size) == input_dim)
                crop_max = [crop_min[i] + bb_max[i] - bb_min[i] \
                    for i in range(input_dim)]
            else:
                crop_max =  [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        image = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        
        sample['image'] = image
        sample['CropWithBoundingBox_Param'] = json.dumps((input_shape, crop_min, crop_max))
        if('label' in sample):
            label = sample['label']
            crop_max[0] = label.shape[0]
            label = crop_ND_volume_with_bounding_box(label, crop_min, crop_max)
            sample['label'] = label

        return sample

    def inverse_transform_for_prediction(self, sample):
        ''' rescale sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        if(isinstance(sample['CropWithBoundingBox_Param'], list) or \
            isinstance(sample['CropWithBoundingBox_Param'], tuple)):
            params = json.loads(sample['CropWithBoundingBox_Param'][0]) 
        else:
            params = json.loads(sample['CropWithBoundingBox_Param']) 
        origin_shape = params[0]
        crop_min     = params[1]
        crop_max     = params[2]
        predict = sample['predict']
        origin_shape   = list(predict.shape[:2]) + origin_shape[1:]
        output_predict = np.zeros(origin_shape, predict.dtype)
        crop_min = [0, 0] + crop_min[1:]
        crop_max = list(predict.shape[:2]) + crop_max[1:]
        output_predict = set_ND_volume_roi_with_bounding_box_range(output_predict,
            crop_min, crop_max, predict)

        sample['predict'] = output_predict
        return sample

class RandomCrop(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W]) 

    Args:
        output_size (tuple or list): Desired output size [D, H, W] or [H, W].
            the output channel is the same as the input channel.
    """

    def __init__(self, output_size, fg_focus = False, fg_ratio = 0.0, mask_label = None,  inverse = True):
        assert isinstance(output_size, (list, tuple))
        if(mask_label is not None):
            assert isinstance(mask_label, (list, tuple))
        self.output_size = output_size
        self.inverse  = inverse
        self.fg_focus = fg_focus
        self.fg_ratio = fg_ratio
        self.mask_label = mask_label
        

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim   = len(input_shape) - 1

        assert(input_dim == len(self.output_size))
        crop_margin = [input_shape[i + 1] - self.output_size[i] for i in range(input_dim)]
        crop_min = [random.randint(0, item) for item in crop_margin]

        crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        
        # do not crop along D axis
        # crop_min[0] = 0
        # crop_max[0] = input_shape[1]

        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        image = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
       
        sample['image'] = image
        sample['RandomCrop_Param'] = json.dumps((input_shape, crop_min, crop_max))
        # if('label' in sample):
        #     label = sample['label']
        #     crop_max[0] = label.shape[0]
        #     label = crop_ND_volume_with_bounding_box(label, crop_min, crop_max)
        #     sample['label'] = label
        if ('gt_map' in sample):
            label = sample['gt_map']
            crop_max[0] = label.shape[0]
            label = crop_ND_volume_with_bounding_box(label, crop_min, crop_max)
            sample['gt_map'] = label
        if('contour' in sample):
            contour = sample['contour']
            crop_max[0] = contour.shape[0]
            contour = crop_ND_volume_with_bounding_box(contour, crop_min, crop_max)
            sample['contour'] = contour
        return sample

    def inverse_transform_for_prediction(self, sample):
        ''' rescale sample['predict'] (5D or 4D) to the original spatial shape.
         assume batch size is 1, otherwise scale may be different for 
         different elemenets in the batch.

        origin_shape is a 4D or 3D vector as saved in __call__().'''
        if(isinstance(sample['RandomCrop_Param'], list) or \
            isinstance(sample['RandomCrop_Param'], tuple)):
            params = json.loads(sample['RandomCrop_Param'][0]) 
        else:
            params = json.loads(sample['RandomCrop_Param']) 
        origin_shape = params[0]
        crop_min     = params[1]
        crop_max     = params[2]
        predict = sample['predict']
        origin_shape   = list(predict.shape[:2]) + origin_shape[1:]
        output_predict = np.zeros(origin_shape, predict.dtype)
        crop_min = [0, 0] + crop_min[1:]
        crop_max = list(predict.shape[:2]) + crop_max[1:]
        output_predict = set_ND_volume_roi_with_bounding_box_range(output_predict,
            crop_min, crop_max, predict)

        sample['predict'] = output_predict
        return sample

class CenterCrop(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W])

    Args:
        output_size (tuple or list): Desired output size [D, H, W] or [H, W].
            the output channel is the same as the input channel.
    """

    def __init__(self, output_size, fg_focus=False, fg_ratio=0.0, mask_label=None, inverse=True):
        assert isinstance(output_size, (list, tuple))
        if (mask_label is not None):
            assert isinstance(mask_label, (list, tuple))
        self.output_size = output_size
        self.inverse = inverse
        self.fg_focus = fg_focus
        self.fg_ratio = fg_ratio
        self.mask_label = mask_label

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim = len(input_shape) - 1

        assert (input_dim == len(self.output_size))
        crop_min = [0, (input_shape[2] - self.output_size[1])//2, (input_shape[3] - self.output_size[2])//2]
        crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        # do not crop along D axis
        crop_min[0] = 0
        crop_max[0] = input_shape[1]

        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        image = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)

        sample['image'] = image
        sample['RandomCrop_Param'] = json.dumps((input_shape, crop_min, crop_max))
        # if ('label' in sample):
        #     label = sample['label']
        #     crop_max[0] = label.shape[0]
        #     label = crop_ND_volume_with_bounding_box(label, crop_min, crop_max)
        #     sample['label'] = label
        if ('gt_map' in sample):
            label = sample['gt_map']
            crop_min_gt = [crop_min[2], crop_min[3]]
            crop_max_gt = [crop_max[2], crop_max[3]]
            label = crop_ND_volume_with_bounding_box(label, crop_min_gt, crop_max_gt)
            sample['gt_map'] = label
        if ('contour' in sample):
            contour = sample['contour']
            crop_max[0] = contour.shape[0]
            contour = crop_ND_volume_with_bounding_box(contour, crop_min, crop_max)
            sample['contour'] = contour
        return sample


class ChannelWiseGammaCorrection(object):
    """
    apply gamma correction to each channel
    """
    def __init__(self, gamma_min, gamma_max, inverse = False):
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.inverse = inverse
    
    def __call__(self, sample):
        image= sample['image']
        for chn in range(image.shape[0]):
            gamma_c = random.random() * (self.gamma_max - self.gamma_min) + self.gamma_min
            img_c = image[chn]
            v_min = img_c.min()
            v_max = img_c.max()
            img_c = (img_c - v_min)/(v_max - v_min)
            img_c = np.power(img_c, gamma_c)*(v_max - v_min) + v_min
            image[chn] = img_c

        sample['image'] = image
        
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class ChannelWiseNormalize(object):
    """Nomralize the image (shape [C, D, H, W] or [C, H, W]) for each channel

    Args:
        mean (None or tuple/list): The mean values along each channel.
        std  (None or tuple/list): The std values along each channel.
            if mean and std are None, calculate them from non-zero region
        chns (None, or tuple/list): The list of channel indices
        zero_to_random (bool, or tuple/list or bool): indicate whether zero values
             in each channel is replaced  with random values.
    """
    def __init__(self, mean, std, chns=None, zero_to_random=False, inverse=False):
        self.mean = mean
        self.std = std
        self.chns = chns
        self.zero_to_random = zero_to_random
        self.inverse = inverse

    def __call__(self, sample):
        image = sample['image']
        mask = image[0] > 0
        chns = self.chns
        if(chns is None):
            chns = range(image.shape[0])
        zero_to_random = self.zero_to_random
        if(isinstance(zero_to_random, bool)):
            zero_to_random = [zero_to_random]*len(chns)
        if(not(self.mean is None and self.std is None)):
            assert(len(self.mean) == len(self.std))
            assert(len(self.mean) == len(chns))
        for i in range(len(chns)):
            chn = chns[i]
            if(self.mean is None and self.std is None):
                pixels = image[chn][mask > 0]
                chn_mean = pixels.mean()
                chn_std  = pixels.std()
            else:
                chn_mean = self.mean[i]
                chn_std  = self.std[i]
            chn_norm = (image[chn] - chn_mean)/chn_std
            if(zero_to_random[i]):
                chn_random = np.random.normal(0, 1, size = chn_norm.shape)
                chn_norm[mask == 0] = chn_random[mask == 0]
            image[chn] = chn_norm

        sample['image'] = image
        return sample

    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class ChannelWiseThreshold(object):
    """Threshold the image (shape [C, D, H, W] or [C, H, W]) for each channel

    Args:
        threshold (tuple/list): The threshold value along each channel.
    """
    def __init__(self, threshold, inverse = False):
        self.threshold = threshold
        self.inverse = inverse

    def __call__(self, sample):
        image= sample['image']
        for chn in range(image.shape[0]):
            mask = np.asarray(image[chn] > self.threshold[chn], image.dtype)
            image[chn] = mask * (image[chn] - self.threshold[chn])

        sample['image'] = image
        return sample

    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class ChannelWiseThresholdWithNormalize(object):
    """Threshold the image (shape [C, D, H, W] or [C, H, W]) for each channel
       and then normalize the image based on remaining pixels

    Args:
        threshold_lower (tuple/list/None): The lower threshold value along each channel.
        threshold_upper (typle/list/None): The upper threshold value along each channel.
        mean_std_mode (bool): If true, nomalize the image based on mean and std values,
            and pixels values outside the threshold value are replaced random number.
            If false, use the min and max values for normalization.
    """
    def __init__(self, threshold_lower, threshold_upper, mean_std_mode = True, inverse = False):
        self.threshold_lower = threshold_lower
        self.threshold_upper = threshold_upper
        self.mean_std_mode   = mean_std_mode
        self.inverse = inverse

    def __call__(self, sample):
        image= sample['image']
        for chn in range(image.shape[0]):
            v0 = self.threshold_lower[chn]
            v1 = self.threshold_upper[chn]
            if(self.mean_std_mode == True):
                mask = np.ones_like(image[chn])
                if(v0 is not None):
                    mask = mask * np.asarray(image[chn] > v0)
                if(v1 is not None):
                    mask = mask * np.asarray(image[chn] < v1)
                pixels   = image[chn][mask > 0]
                chn_mean = pixels.mean()
                chn_std  = pixels.std()
                chn_norm = (image[chn] - chn_mean)/chn_std
                chn_random = np.random.normal(0, 1, size = chn_norm.shape)
                chn_norm[mask == 0] = chn_random[mask == 0]
                image[chn] = chn_norm
            else:
                img_chn = image[chn]
                if(v0 is not None):
                    img_chn[img_chn < v0] = v0
                    min_value = v0 
                else:
                    min_value = img_chn.min()
                if(v1 is not None):
                    img_chn[img_chn > v1] = v1 
                    max_value = img_chn.max() 
                else:
                    max_value = img_chn.max() 
                img_chn = (img_chn - min_value) / (max_value - min_value)
                image[chn] = img_chn
        sample['image'] = image
        return sample

class ReduceLabelDim(object):
    def __init__(self, inverse = False):
        self.inverse = inverse
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = label[0]
        sample['label'] = label_converted
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class LabelConvert(object):
    """ Convert a list of labels to another list
    Args:
        source_list (tuple/list): A list of labels to be converted
        target_list (tuple/list): The target label list
    """
    def __init__(self, source_list, target_list, inverse = False):
        self.source_list = source_list
        self.target_list = target_list
        self.inverse = inverse
        assert(len(source_list) == len(target_list))
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = convert_label(label, self.source_list, self.target_list)
        sample['label'] = label_converted
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class LabelConvertNonzero(object):
    """ Convert a list of labels to another list
    Args:
        source_list (tuple/list): A list of labels to be converted
        target_list (tuple/list): The target label list
    """
    def __init__(self, inverse = False):
        self.inverse = inverse
    
    def __call__(self, sample):
        label = sample['label']
        label_converted = np.asarray(label > 0, np.uint8)
        sample['label'] = label_converted
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class LabelToProbability(object):
    """
        Convert one-channel label map to multi-channel probability map
    Args:
        class_num (int): the class number in the label map
    """
    def __init__(self, class_num, inverse = False):
        self.class_num = class_num
        self.inverse   = inverse
    
    def __call__(self, sample):
        label = sample['label'][0]
        label_prob = []
        for i in range(self.class_num):
            temp_prob = label == i*np.ones_like(label)
            label_prob.append(temp_prob)
        label_prob = np.asarray(label_prob, np.float32)
        sample['label_prob'] = label_prob
        return sample
    
    def inverse_transform_for_prediction(self, sample):
        raise(ValueError("not implemented"))

class ProbabilityToDistance(object):
    """
     get distance transform for each label
    """
    def __init__(self, inverse = False):
        self.inverse = inverse

    
    def __call__(self, sample):
        label_prob = sample['label_prob']
        label_distance = []
        for i in range(label_prob.shape[0]):
            temp_lab = label_prob[i]
            temp_dis = get_euclidean_distance(temp_lab, dim = 3, spacing = [1.0, 1.0, 1.0])
            label_distance.append(temp_dis)
        label_distance = np.asarray(label_distance)
        sample['label_distance'] = label_distance
        return sample

class RegionSwop(object):
    """
    Swop a subregion randomly between two images and their corresponding label
    Args:
        axes: the list of possible specifed spatial axis for swop, 
              if None, then it is all the spatial axes
        prob: the possibility of use region swop

    """
    def __init__(self, spatial_axes = None, probility = 0.5, inverse = False):
        self.axes = spatial_axes
        self.prob = probility
        self.inverse = inverse
    
    def __call__(self, sample):
        # image shape is like [B, C, D, H, W]
        img = sample['image']
        [B, C, D, H, W] = img.shape
        if(B < 2):
            return sample
        swop_flag = random.random() < self.prob
        if(swop_flag):
            swop_axis = random.sample(self.axes, 1)[0]
            ratio = random.random()
            roi_min = [0, 0, 0, 0]
            if(swop_axis == 0):
                d = int(D*ratio)
                roi_max = [C, d, H, W]
            elif(swop_axis == 1):
                h = int(H*ratio)
                roi_max = [C, D, h, W]
            else:
                w = int(W*ratio)
                roi_max = [C, D, H, w]
            img_sub0 = crop_ND_volume_with_bounding_box(img[0], roi_min, roi_max)
            img_sub1 = crop_ND_volume_with_bounding_box(img[1], roi_min, roi_max)
            img[0] = set_ND_volume_roi_with_bounding_box_range(img[0], roi_min, roi_max, img_sub1)
            img[1] = set_ND_volume_roi_with_bounding_box_range(img[1], roi_min, roi_max, img_sub0)
            sample['image'] = img
            if('label' in sample):
                label = sample['label']
                roi_max[0] = label.shape[1]
                lab_sub0 = crop_ND_volume_with_bounding_box(label[0], roi_min, roi_max)
                lab_sub1 = crop_ND_volume_with_bounding_box(label[1], roi_min, roi_max)
                label[0] = set_ND_volume_roi_with_bounding_box_range(label[0], roi_min, roi_max, lab_sub1)
                label[1] = set_ND_volume_roi_with_bounding_box_range(label[1], roi_min, roi_max, lab_sub0)
                sample['label'] = label
        return sample

