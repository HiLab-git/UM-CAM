import itertools
import os
import random
import SimpleITK as sitk
import cv2
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


class BaseDatasets_fetal(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, image='image', sup='pseudo'):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.image = image
        self.sample_list = os.listdir('{}/{}/{}'.format(os.path.dirname(self._base_dir), self.image, self.split))
        self.sup = sup

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        filename = os.path.join('{}/{}/{}/{}'.format(os.path.dirname(self._base_dir), self.image, self.split, case))
        if self.split == 'train':
            image = sitk.GetArrayFromImage(sitk.ReadImage(filename)).squeeze()
            label = sitk.GetArrayFromImage(sitk.ReadImage(filename.replace(self.image, self.sup))).squeeze()
            image = (image - image.min()) / (image.max() - image.min())
            sample = {'image': image, 'label': label}
            if self.transform is not None:
                sample = self.transform(sample)
            else:
                image = image / 1.0
                image = torch.FloatTensor(image)
                label = label.astype(np.uint8)
                sample['image'] = image
                sample['label'] = label
            sample['idx'] = idx
            sample['filename'] = case
        else:
            image = sitk.GetArrayFromImage(sitk.ReadImage(filename)).squeeze()
            label = sitk.GetArrayFromImage(sitk.ReadImage(filename.replace('image', 'label'))).squeeze()

            image = (image - image.min()) / (image.max() - image.min())
            image = image / 1.0
            image = torch.FloatTensor(image)
            
            sample = {'image': image, 'label': label, 'idx': idx, 'filename': case}
        return sample



class Dataset_fetal_prob(Dataset):
    def __init__(self, base_dir=None, split='train', transform=None, image='image', sup="pseudo"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.image = image
        self.sample_list = os.listdir('{}/{}/{}'.format(os.path.dirname(self._base_dir), self.image, self.split))
        self.sup = sup
        self.sharpen = lambda p,T: p**(1.0/T)/(p**(1.0/T) + (1-p)**(1.0/T))

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        filename = os.path.join('{}/{}/{}/{}'.format(os.path.dirname(self._base_dir), self.image, self.split, case))
        if self.split == 'train':
            image = sitk.GetArrayFromImage(sitk.ReadImage(filename)).squeeze()
            image = (image - image.min()) / (image.max() - image.min())

            pseudo_name = os.path.join(os.path.dirname(self._base_dir), self.sup, self.split, case)
            label = sitk.GetArrayFromImage(sitk.ReadImage(pseudo_name)).squeeze()

            sample = {'image': image, 'label': label}
            if self.transform is not None:
                sample = self.transform(sample)
            label = sample['label']
            sample['label'] = torch.concat([(1-label).unsqueeze(0), label.unsqueeze(0)], 0)
            sample['idx'] = idx
            sample['filename'] = case
        else:
            image = sitk.GetArrayFromImage(sitk.ReadImage(filename)).squeeze()
            image = (image - image.min()) / (image.max() - image.min())
            image = image / 1.0
            image = torch.FloatTensor(image)

            label = sitk.GetArrayFromImage(sitk.ReadImage(filename.replace('image', 'label'))).squeeze()
            sample = {'image': image, 'label': label, 'idx': idx, 'filename': case}
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode="constant", cval=cval)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label = random_rotate(image, label, cval=4)
            else:
                image, label = random_rotate(image, label, cval=0)

        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label}

        return sample
