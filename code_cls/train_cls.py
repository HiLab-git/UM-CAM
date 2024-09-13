from tensorboardX import SummaryWriter
from code_cls.utils.util import *
from code_cls.utils.dataset import train_dataloader, test_dataloader
from network.vgg import vgg16, vgg16_DRS
import sys
import os
import torch
import argparse
import torch.optim as optim
import logging
import datetime
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch import nn

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])


def config():
    parser = argparse.ArgumentParser(description='classification implementation')
    parser.add_argument("--root_path", type=str, default='/home/data/FJ/Dataset/Fetal_brain/Fetal_weak')
    parser.add_argument("--split", type=str, default="fold0")
    parser.add_argument("--positive_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)    
    parser.add_argument("--model_name", type=str, default="vgg16", choices=['vgg16', 'vgg16_DRS'])
    parser.add_argument("--dataset_name", type=str, default="fetal_brain")
    parser.add_argument('--patch_size', type=int,  default=224, help='patch size of network input')
    parser.add_argument('--seed', type=int,  default=2023, help='random seed')
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--save_folder', default='Code/code_cls/result/checkpoints', help='Location to save checkpoint models')
    parser.add_argument('--show_interval', default=50, type=int, help='interval of showing training conditions')
    parser.add_argument('--save_interval', default=10, type=int, help='interval of save checkpoint models')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.55, help='set 0 for the learnable DRS')
    parser.add_argument("--alpha", type=float, default=0.00, help='object cues for the pseudo seg map generation')
    parser.add_argument('--cuda_device', type=str, default="0")
    
    return parser.parse_args()


def get_model(args):
    """vgg16"""
    if args.model_name == "vgg16":
        model = vgg16(pretrained=True, in_channels=3)
        model = torch.nn.DataParallel(model).cuda()
        params = model.module.parameters()
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        args.save_folder = os.path.join(args.save_folder, "vgg16")

    elif args.model_name == "vgg16_DRS":
        model = vgg16_DRS(pretrained=True, delta=args.delta)
        model = torch.nn.DataParallel(model).cuda()
        param_groups = model.module.get_parameter_groups()

        optimizer = optim.SGD([
            {'params': param_groups[0], 'lr': args.lr},
            {'params': param_groups[1], 'lr': 2*args.lr},
            {'params': param_groups[2], 'lr': 10*args.lr},
            {'params': param_groups[3], 'lr': 20*args.lr}],
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True
        )
        args.save_folder = os.path.join(args.save_folder, "vgg16_DRS")
        print(model)
    return model, optimizer


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(current_epoch, loss_func):
    print('\nvalidating ... ', flush=True, end='')
    val_loss = AverageMeter()

    model.eval()
    val_label, val_logit = [], []
    with torch.no_grad():
        for idx, data in tqdm(enumerate(val_loader)):
            img, label = data['image'], data['label']
            img = img.to('cuda', non_blocking=True)
            label = label.to('cuda')

            logit = model(img)

            """ classification loss """
            loss = loss_func(logit, label[:].long())

            """evaluate metrics"""
            logit_argmax = torch.argmax(logit, dim=1, keepdim=True)

            val_label.append(label.data.item())
            val_logit.append(logit_argmax.data.item())
            val_loss.update(loss, img.size()[0])

    accuracy = accuracy_score(val_label, val_logit)
    precision = precision_score(val_label, val_logit, zero_division=0)
    recall = recall_score(val_label, val_logit, zero_division=0)
    f1 = f1_score(val_label, val_logit, zero_division=0)

    """ tensorboard visualization """
    writer.add_scalar('valid/loss', val_loss.avg, current_epoch)
    writer.add_scalar('valid/accuracy', accuracy, current_epoch)
    writer.add_scalar('valid/precision', precision, current_epoch)
    writer.add_scalar('valid/recall', recall, current_epoch)
    writer.add_scalar('valid/f1', f1, current_epoch)
    logging.debug("validating loss {:.4f}, accuracy {:.4f}, precision {:.4f}, recall {:.4f}, f1-score {:.4f}\n".format(val_loss.avg, accuracy, precision, recall, f1))

    return f1


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(current_epoch, loss_func):
    model.train()
    global_counter = args.global_counter
    train_loss = AverageMeter()
    train_accuracy, train_precision, train_recall, train_f1 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for idx, data in enumerate(train_loader):
        img, label = data['image'], data['label']
        img = img.to('cuda', non_blocking=True)
        label = label.to('cuda', non_blocking=True)

        """general training"""
        logit = model(img)
        loss = loss_func(logit, label[:].long())

        """ backprop """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """evaluate metrics"""
        logit_argmax = torch.argmax(logit, dim=1, keepdim=True)
        logit_argmax = logit_argmax.detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        train_loss.update(loss.data.item(), img.size()[0])

        """calculate and save the metrics for each batch"""
        accuracy = accuracy_score(label, logit_argmax)
        precision = precision_score(label, logit_argmax, zero_division=0)
        recall = recall_score(label, logit_argmax, zero_division=0)
        f1 = f1_score(label, logit_argmax, zero_division=0)
        train_accuracy.update(accuracy, img.size()[0])
        train_precision.update(precision, img.size()[0])
        train_recall.update(recall, img.size()[0])
        train_f1.update(f1, img.size()[0])

        global_counter += 1
        """ tensorboard log """
        if global_counter % args.show_interval == 0:
            writer.add_scalar('train/loss', train_loss.avg, global_counter)
            writer.add_scalar('train/acc', train_accuracy.avg, global_counter)

    args.global_counter = global_counter
    logging.debug("epoch {}, train loss {:.4f}, accuracy {:.4f}, precision {:.4f}, recall {:.4f}, f1-score {:.4f}".format(current_epoch, train_loss.avg, train_accuracy.avg, train_precision.avg, train_recall.avg, train_f1.avg))


if __name__ == '__main__':
    args = config()
    args.root_path = "{}/slice_resample_set_{}/image".format(args.root_path, args.split)
    args.dataset_name = "{}_{}".format(args.dataset_name, args.split)

    """generate save folder and log file"""
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')    
    save_path = os.path.join(args.save_folder, args.dataset_name + '_' + args.model_name + '_' + str(nowTime))
    writer = SummaryWriter(log_dir=save_path)
    logging.basicConfig(format="%(asctime)s | %(message)s",
                        datefmt="%Y-%m-%d %H-%M-%S",
                        filename=os.path.join(save_path, "train.log"),
                        level=logging.DEBUG)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = train_dataloader(args, worker_init_fn, num_pos=args.positive_samples)
    val_loader = test_dataloader(args, split='valid')
    model, optimizer = get_model(args)

    print('Running parameters:\n', args)
    print('# of train dataset:', len(train_loader) * args.batch_size)
    print('# of valid dataset:', len(val_loader) * 1)
    print()

    logging.debug('Running parameters: {}'.format(args))
    logging.debug('train dataset: {}'.format(len(train_loader) * args.batch_size))
    logging.debug('valid dataset: {}'.format(len(val_loader) * 1))
    logging.debug("")

    loss_func = nn.CrossEntropyLoss()
    validate(0, loss_func)

    best_score = 0
    for current_epoch in range(1, args.epoch+1):
        train(current_epoch, loss_func)
        score = validate(current_epoch, loss_func)

        """ save checkpoint """
        state = {
            'model': model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            'epoch': current_epoch,
            'iter': args.global_counter,
            'cls accuracy': score,
        }

        if score > best_score:
            best_score = score
            print('\nSaving state, epoch : %d , cls accuracy : %.4f \n' % (current_epoch, score))
            logging.debug('\nSaving state, epoch : %d , cls accuracy : %.4f \n' % (current_epoch, score))
            model_file = os.path.join(save_path, str(current_epoch) + '_best.pth')
        else:
            model_file = os.path.join(save_path, 'latest.pth')

        torch.save(state, model_file)
