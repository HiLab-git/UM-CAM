import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from code_seg.dataloaders.dataset import *
from networks.net_factory import net_factory
from val_2D import test_single_volume

def get_rampup_ratio(i, start, end, mode = "linear"):
    """
    Obtain the rampup ratio.
    :param i: (int) The current iteration.
    :param start: (int) The start iteration.
    :param end: (int) The end itertation.
    :param mode: (str) Valid values are {`linear`, `sigmoid`, `cosine`}.
    """
    i = np.clip(i, start, end)
    if(mode == "linear"):
        rampup = (i - start) / (end - start)
    elif(mode == "sigmoid"):
        phase  = 1.0 - (i - start) / (end - start)
        rampup = float(np.exp(-5.0 * phase * phase))
    elif(mode == "cosine"):
        phase  = 1.0 - (i - start) / (end - start)
        rampup = float(.5 * (np.cos(np.pi * phase) + 1))
    else:
        raise ValueError("Undefined rampup mode {0:}".format(mode))
    return rampup
   
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


def CELoss(predict, soft_y):
    predict = predict * 0.999 + 5e-4
    ce = - soft_y* torch.log(predict)
    ce = torch.sum(ce, dim = 1)
    ce = torch.mean(ce)
    return ce

def CELoss_weight(predict, soft_y, weight):
    predict = predict * 0.999 + 5e-4
    ce = - soft_y* torch.log(predict)
    ce = torch.sum(weight * ce) / (weight.sum() + 1e-5)     
    return ce

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    db_train = Dataset_fetal_prob(base_dir=args.root_path,
                                split="train",
                                transform=transforms.Compose([RandomGenerator(args.patch_size)]),
                                image=args.image,
                                sup=args.sup)
    db_val = Dataset_fetal_prob(base_dir=args.root_path_val,
                                split="valid",
                                transform=None) 
                                    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    g = torch.Generator()
    g.manual_seed(args.seed)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True, generator=g)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.99, weight_decay=0.0005)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_iterations = max_epoch * len(trainloader)
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    scale_factors = [0.5, 0.75, 1.0, 1.5]
    upsample_layers = [torch.nn.Upsample(scale_factor=(s, s), mode="nearest") for s in scale_factors]
    downsample_layers = [torch.nn.Upsample(scale_factor=(1/s, 1/s), mode="nearest") for s in scale_factors]
    
    for epoch_num in iterator:
        rampup_ratio = get_rampup_ratio(epoch_num, args.rampup_start, args.rampup_end)
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
            _, _, h, w = volume_batch.shape
            idx1 = np.random.randint(0, 2)
            idx2 = np.random.randint(0, 2)
            
            def apply_transformations(batch, idx):
                if idx == 0:
                    scale_idx_x = np.random.randint(0, len(scale_factors))
                    upsample = upsample_layers[scale_idx_x]
                    downsample = downsample_layers[scale_idx_x]
                    batch_scaling = upsample(batch)
                    outputs_aug = model(batch_scaling)
                    return downsample(outputs_aug)
                elif idx == 1:
                    rot = np.random.randint(1, 3)
                    batch_rot = torch.rot90(batch, rot, dims=[2, 3])
                    outputs_rot = model(batch_rot)
                    return torch.rot90(outputs_rot, -rot, dims=[2, 3])

            outputs = apply_transformations(volume_batch, idx1)
            outputs_aug = apply_transformations(volume_batch, idx2)

            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_aug_soft = torch.softmax(outputs_aug, dim=1)
            
            """two outputs weighted CE loss"""
            if args.loss_weighting:
                outputs_soft_mean = (outputs_soft + outputs_aug_soft) / 2
                uncertainty_map = (outputs_soft - outputs_soft_mean) ** 2
                uncertainty_map_scaling = (outputs_aug_soft - outputs_soft_mean) ** 2
                weight_map = 1 - uncertainty_map
                weight_map_scaling = 1 - uncertainty_map_scaling
                
                loss_ce1 = CELoss_weight(outputs_soft, label_batch, weight_map) 
                loss_ce2 = CELoss_weight(outputs_aug_soft, label_batch, weight_map_scaling) 
            else:
                loss_ce1 = CELoss(outputs_soft, label_batch) 
                loss_ce2 = CELoss(outputs_aug_soft, label_batch) 
            loss_ce = torch.mean((loss_ce1 + loss_ce2) / 2)

            probs = [(F.softmax(logits, dim=1) + 1e-6) for i, logits in enumerate([outputs, outputs_aug])]
            consistency = F.kl_div(probs[1].log(), probs[0], reduction='mean')
            consistency += F.kl_div(probs[0].log(), probs[1], reduction='mean')
            loss_cons = torch.mean(consistency)

            loss = loss_ce + args.consistency_weight * rampup_ratio * loss_cons
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            logging.info('iteration %d : loss: %f, loss_ce: %f, loss_con: %f' % (iter_num, loss.item(), loss_ce.item(), loss_cons.item()))
            
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='fetal', help='experiment_name')
    parser.add_argument('--sup_type', type=str, default='image_level', help='supervision type')
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--root_path_val', type=str, default='')
    parser.add_argument('--image', type=str, default='image_seg')
    parser.add_argument('--sup', type=str, default='pseudo_hard')
    parser.add_argument('--loss_type', type=str, default='ours', help='loss type')
    parser.add_argument('--model', type=str, default='unet', help='model_name')
    parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
    parser.add_argument('--max_epoch', type=int, default=200, help='maximum epoch number to train')    
    parser.add_argument('--batch_size', type=int, default=12, help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
    parser.add_argument('--seed', type=int,  default=2022, help='random seed')
    parser.add_argument('--consistency_weight', type=float, default=0.1, help='consistency weight')
    parser.add_argument('--rampup_start', type = int, default = 20)
    parser.add_argument('--rampup_end', type = int, default = 200)
    parser.add_argument('--cuda_device', type=str,  default="0")
    parser.add_argument("--loss_weighting", type=str2bool, default=False)
    
    args = parser.parse_args()
    
    return args



if __name__ == "__main__":
    args = argument()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    
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

    snapshot_path = "code_seg/model/{}/{}".format(args.exp, args.loss_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
