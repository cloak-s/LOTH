import os
import sys
import errno
import shutil
import os.path as osp
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch.autograd import Variable
import pdb

import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))

# def load_checkpoint(model, checkpoint):
#     m_keys = list(model.state_dict().keys())
#
#     if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#         c_keys = list(checkpoint['state_dict'].keys())
#         not_m_keys = [i for i in c_keys if i not in m_keys]
#         not_c_keys = [i for i in m_keys if i not in c_keys]
#         model.load_state_dict(checkpoint['state_dict'], strict=False)
#
#     else:
#         c_keys = list(checkpoint.keys())
#         not_m_keys = [i for i in c_keys if i not in m_keys]
#         not_c_keys = [i for i in m_keys if i not in c_keys]
#         model.load_state_dict(checkpoint, strict=False)
#
#     print("--------------------------------------\n LOADING PRETRAINING \n")
#     print("Not in Model: ")
#     print(not_m_keys)
#     print("Not in Checkpoint")
#     print(not_c_keys)
#     print('\n\n')


from typing import Any
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
from PIL import Image


class TrainTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None) -> None:
        super().__init__()
        self.filenames = glob.glob(root + "/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: Any) -> Any:
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.id_dict[img_path.split('/')[-3]]
        if self.transform:
            image = self.transform(image)
        return image, label


class ValTinyImageNet(Dataset):
    def __init__(self, root, id, transform=None):
        self.filenames = glob.glob(root + "/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open(root + '/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.id_dict[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_tinyimagenet(args):
    batch_size = args.batch_size
    nw = args.workers
    root = args.data_dir
    id_dic = {}
    for i, line in enumerate(open(root + '/wnids.txt', 'r')): # '\wnids.txt', 'r'
        id_dic[line.replace('\n', '')] = i
    num_classes = len(id_dic)
    data_transform = {
        "train": transforms.Compose([transforms.Resize(32),  # 224
                                     transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(32),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_dataset = TrainTinyImageNet(root, id=id_dic, transform=data_transform["train"])
    val_dataset = ValTinyImageNet(root, id=id_dic, transform=data_transform["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw)

    print("TinyImageNet Loading SUCCESS" +
          "\nlen of train dataset: " + str(len(train_dataset)) +
          "\nlen of val dataset: " + str(len(val_dataset)) +
          "\nlen of image class:" + str(num_classes))

    return train_loader, val_loader, num_classes


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def trainsform(mean, std):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return transform_train, transform_test


def load_cifar(args):
    # just train and test data
    data_name = os.path.basename(args.data_dir)
    if data_name == 'CIFAR10':
        print('DATA:', data_name)
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        transform_train, transform_test = trainsform(cifar10_mean, cifar10_std)
        train_loader = DataLoader(
            datasets.CIFAR10(root=args.data_dir, train=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )

        test_loader = DataLoader(
            datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )
        num_classes = 10
        return train_loader, test_loader, num_classes

    elif data_name == 'CIFAR100':
        print('DATA:', data_name)
        cifar100_mean = (0.5071, 0.4867, 0.4408)
        cifar100_std = (0.2675, 0.2565, 0.2761)
        transform_train, transform_test = trainsform(cifar100_mean, cifar100_std)
        train_loader = DataLoader(
            datasets.CIFAR100(root=args.data_dir, train=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )

        test_loader = DataLoader(
            datasets.CIFAR100(root=args.data_dir, train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )
        num_classes = 100
        return train_loader, test_loader, num_classes


class KLLoss(nn.Module):
    def __init__(self, temperature):
        super(KLLoss, self).__init__()
        self.tem = temperature

    def forward(self, pred, label):
        predict = F.log_softmax(pred / self.tem, dim=1)
        target_data = F.softmax(label / self.tem, dim=1)
        target_data = target_data + 10 ** (-7)
        target = Variable(target_data.data.cuda(), requires_grad=False) #
        loss = self.tem * self.tem * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        return loss


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def lossOfKnowledge(lossFunction, crossFusionKnowledge):
    return sum([lossFunction(knowledgePair[0], knowledgePair[1]) + lossFunction(knowledgePair[1], knowledgePair[0]) for
                knowledgePair in crossFusionKnowledge])


def accuracy(output, target, topk=(1,5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def random_amp(t_feat, s_c):
    # default teacher channel is max
    b, t_c, h, w = t_feat.shape
    # _, s_c, _, _ = s_feat.shape

    num = t_c // s_c
    mapping = torch.randperm(t_c)

    group_feat = []
    for c in range(0, t_c, s_c):
        group_feat.append(t_feat[:, mapping[c:c + s_c], :, :])
    new_t_feat = torch.stack(group_feat, dim=2)
    new_t_feat = new_t_feat.reshape(b, s_c, num, -1)

    t_inds = torch.argmax(torch.abs(new_t_feat), dim=2)
    new_t_feat = new_t_feat.gather(2, t_inds.unsqueeze(2))

    new_t_feat = new_t_feat.reshape(b, s_c, h, w)
    return new_t_feat




