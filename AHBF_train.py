from utils import AverageMeter, accuracy, load_cifar, KLLoss, set_seed, sigmoid_rampup, load_tinyimagenet
import torch
import torch.nn as nn
from models import AHBF_ResNet18, KL_Loss
from Backbone import Res_18
import argparse
import json
import time
from datetime import datetime
import warnings
import os
import gc
import torch.optim as optim


warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda")
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 training')
parser.add_argument('--data-dir', default='./data/CIFAR100', type=str, help='the diretory to save cifar100 dataset')
# parser.add_argument('--data-dir', default='./data/tiny-imagenet-200', type=str, help='the diretory to save cifar100 dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4 )')
parser.add_argument('--start-epoch', default=0, type=int, help='manual iter number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--epochs', default=200, type=int, help='number of total iterations (default:300)')
parser.add_argument('--milestones', default=[75, 130, 180], nargs="*", type=list)
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--model_name', default='Res18', type=str)
parser.add_argument('--resume', default='', type=str, help='path to  latest checkpoint (default: None)')
parser.add_argument('--save-folder', default='AHBF/', type=str, help='folder to save the checkpoints')
parser.add_argument('--cuda_num', default='0', type=str)
parser.add_argument('--temperature', default=3.0, type=int, help='temperature to smooth the logits')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--consistency_rampup', '--consistency_rampup', default=80, type=float,
                    metavar='consistency_rampup', help='consistency_rampup ratio')
parser.add_argument('--num_branches', default=3, type=int, help='numbers of branch')
parser.add_argument('--lambda2', default=2.0, type=float)  # 论文里是 4
parser.add_argument('--lambda1', default=1.0, type=float)   # 2
parser.add_argument('--kd_weight', default=1.0, type=float)


args = parser.parse_args()
Criterion_CE = nn.CrossEntropyLoss()
Criterion_KD = KL_Loss(args.temperature)
Criterion_MSE = nn.MSELoss()


def AHBF_train(epoch, model, optimizer, train_loader):
    print('\nEPOCH: %d' % epoch)
    train_loss = AverageMeter('Train_Loss', ':.4e')
    Acc_main = AverageMeter('Acc@1', ':6.2f')
    Acc_ensemble = AverageMeter('Acc@2', ':6.2f')
    train_start_time = time.time()

    model.train()
    consistent_weight = sigmoid_rampup(epoch, args.consistency_rampup)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        logit_list, ensemble_list = model(inputs)

        loss_true = 0
        loss_group_ekd = 0
        loss_group_dkd = 0
        for logit in logit_list:
            loss_true += Criterion_CE(logit, targets)
        for ensemble in ensemble_list:
            loss_true += Criterion_CE(ensemble, targets)
        for i in range(0, args.num_branches - 1):
            if i == 0:
                loss_group_dkd += Criterion_KD(logit_list[i], logit_list[i + 1]) * args.lambda2
                loss_group_ekd += Criterion_KD(logit_list[i], ensemble_list[i]) * args.lambda1
                loss_group_ekd += Criterion_KD(logit_list[i + 1], ensemble_list[i]) * args.lambda1
            else:
                loss_group_dkd += Criterion_KD(ensemble_list[i-1], logit_list[i + 1]) * args.lambda2
                loss_group_ekd += Criterion_KD(logit_list[i+1], ensemble_list[i]) * args.lambda1
                loss_group_ekd += Criterion_KD(ensemble_list[i - 1], ensemble_list[i]) * args.lambda1

        total_loss = loss_true + consistent_weight * args.kd_weight * (loss_group_dkd + loss_group_ekd)

        train_loss.update(total_loss, inputs.size(0))

        prec1 = accuracy(logit_list[0].data, targets, topk=(1,))
        Acc_main.update(prec1[0], inputs.size(0))

        prec2 = accuracy(ensemble_list[-1].data, targets, topk=(1,))
        Acc_ensemble.update(prec2[0], inputs.size(0))

        # updating
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print('Training ** \t Time Taken: %.2f sec' % (time.time() - train_start_time))
    print('Loss: %.3f | Acc@main: %.3f%% | Acc@ensemble: %.3f%% |' %
          (train_loss.avg, Acc_main.avg, Acc_ensemble.avg))


def AHBF_val(model, test_loader):
    val_loss = AverageMeter('Loss', ':.4e')
    Acc_main = AverageMeter('Acc@1', ':6.2f')
    Acc_ensemble = AverageMeter('Acc@2', ':6.2f')
    val_start_time = time.time()
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            logit_list, ensemble_list = model(inputs)
            total_loss = Criterion_CE(logit_list[0], targets)

            ensemble_logit = sum(logit_list) / len(logit_list)

            val_loss.update(total_loss, inputs.size(0))
            prec1 = accuracy(logit_list[0].data, targets, topk=(1,))
            Acc_main.update(prec1[0], inputs.size(0))
            prec2 = accuracy(ensemble_logit.data, targets, topk=(1,))
            Acc_ensemble.update(prec2[0], inputs.size(0))
    print('Training ** \t Time Taken: %.2f sec' % (time.time() - val_start_time))
    print('Loss: %.3f | Acc@main: %.3f%% | Acc@ensemble: %.3f%% |' %
          (val_loss.avg, Acc_main.avg, Acc_ensemble.avg))

    return Acc_main.avg, Acc_ensemble.avg



if __name__ == '__main__':
    print(args)
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    train_loader, test_loader, num_class = load_cifar(args)
    model = AHBF_ResNet18(num_classes=num_class, num_branches=args.num_branches, aux=2).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    time_log = datetime.now().strftime('%m_%d_%H-%M')
    folder_name = '{}_{}'.format(args.model_name, time_log)
    path = os.path.join(args.save_folder, folder_name)
    path = path.replace('\\', '/')

    if not os.path.exists('logs/' + path):
        os.makedirs('logs/' + path)
    with open('logs/{}/parameters.txt'.format(path), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    max_acc1 = 0
    max_epoch = 0

    max_fuse = 0
    max_f_epoch = 0

    for epoch in range(1, args.epochs + 1):
        f = open(os.path.join("logs/" + path, 'log.txt'), "a")

        AHBF_train(epoch, model, optimizer, train_loader)
        acc_main, acc_ensemble = AHBF_val(model, test_loader)
        scheduler.step()

        if max_acc1 <= acc_main:
            max_acc1 = acc_main
            max_epoch = epoch

        if max_fuse <= acc_ensemble:
            max_fuse = acc_ensemble
            max_f_epoch = epoch

        f.write('EPOCH:{EPOCH} \t Exit@main:{exit_main:.4f} \t Exit@ensemble:{exit_ensemble:.4f} \t '
                'BEST_Epoch: {best_epoch} \t Max_main: {max_main:.4f} \t'
                'Max_ensemble:{max_ensemble:.4f} \n'.format(EPOCH=epoch,
                                                            exit_main=acc_main.item(),
                                                            exit_ensemble=acc_ensemble.item(),
                                                            best_epoch=max_epoch, max_main=max_acc1.item(),
                                                            max_ensemble=max_fuse.item()))

        print('BEST *** acc@main: {:.3f}% *** main_epoch:{} *** acc@ensemble: {:.3f} *** ensemble_epoch:{} '
              .format(max_acc1.item(), max_epoch, max_fuse.item(), max_f_epoch))
        f.close()

        gc.collect()