import os
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from .transforms import transforms
from .transform3d import *
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

class WSLDataset(Dataset):
    def __init__(self, base_dir, split='train', transform=None, balanced=False, num_pos=None):
        super().__init__()
        self._base_dir = base_dir
        self.split = split
        self.transform = transform
        self.balanced = balanced
        self.num_pos = num_pos
        self.sample_list = self.__get_sample_list__()
        self.csv_label = 'cls_label.csv'
        self.df = pd.read_csv(self.csv_label)

    def __get_sample_list__(self):
        if (self.balanced) and (self.split == "train"):
            assert (self.num_pos > 0)
            filenames = os.listdir(self._base_dir + "/" + self.split)
            random.shuffle(filenames)
            sample_list = []
            n_neg = 0
            for i in range(len(filenames)):
                case = filenames[i]
                filename = os.path.join(self._base_dir, self.split, case)
                gt_map = sitk.GetArrayFromImage(sitk.ReadImage(filename.replace('image', 'label')))
                if gt_map.sum() == 0:                    
                    if n_neg < self.num_pos:
                        n_neg += 1
                        sample_list.append(case)
                else:
                    sample_list.append(case)
        else:
            sample_list = os.listdir(self._base_dir + "/" + self.split)
        
        return sample_list

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        filename = os.path.join(self._base_dir, self.split, case)
        image = sitk.GetArrayFromImage(sitk.ReadImage(filename)).squeeze()
        image = (image - image.min()) / (image.max() - image.min())
        image = np.expand_dims(image, 0).astype('float32')
        image = np.repeat(image, 3, axis=0)

        sample = {'image': image}
        if self.transform is not None:
            sample = self.transform(sample)
        
        idx_case = self.df['filename'].index(case.split('.')[0])
        label = self.df['label'][idx_case]
        
        sample['label'] = label
        sample['filename'] = case
        
        return sample



def train_dataloader(args, worker_init_fn, split='train', num_pos=None):
    balanced = args.balanced_dataset
    num_pos = num_pos
    
    tsfm_train = Compose([RandomFlip(flip_depth=False, flip_height=True, flip_width=True, inverse=True),
                          RandomRotate(angle_range_d=[-30, 30], angle_range_h=None, angle_range_w=None, inverse=False),
                          Rescale([256, 256]),
                          RandomCrop([args.patch_size, args.patch_size])
                          ])
    

    img_train = WSLDataset(base_dir=args.root_path, split=split, transform=tsfm_train, balanced=balanced, num_pos=num_pos)
    train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn, drop_last=True)
    
    return train_loader

def test_dataloader(args, split='test'):
    tsfm_test = Compose([Rescale([args.patch_size, args.patch_size])])
    
    img_test = WSLDataset(base_dir=args.root_path, split=split, transform=tsfm_test)
    train_loader = DataLoader(img_test, batch_size=1, num_workers=args.num_workers)
    
    return train_loader