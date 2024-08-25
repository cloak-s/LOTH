from utils import AverageMeter, accuracy, load_cifar, KLLoss, set_seed, sigmoid_rampup, intra_fd, load_tinyimagenet
import torch
# from Process_Image import load_tinyimagenet
import torch.nn as nn
from Backbone import Res_18, Res_34, Norm_fusion, MobileNetV2, VGG13_bn, resnext50_32x4d, ShuffleNet
from models import ShuffleNetV2, MobileNetV1, VGG16_bn, VGG19_bn, MobileNetV2, ShuffleNet, Res_50
import argparse
import json
import time
from datetime import datetime
import warnings
import os
import gc
import torch.optim as optim
import torch.nn.functional as F


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
parser.add_argument('--model_name', default='Res50', type=str)
parser.add_argument('--resume', default='', type=str, help='path to  latest checkpoint (default: None)')
parser.add_argument('--save-folder', default='Baseline/', type=str, help='folder to save the checkpoints')
parser.add_argument('--cuda_num', default='0', type=str)
parser.add_argument('--temperature', default=3.0, type=int, help='temperature to smooth the logits')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--consistency_rampup', '--consistency_rampup', default=20, type=float,
                    metavar='consistency_rampup', help='consistency_rampup ratio')
parser.add_argument('--alpha', default=1.15, type=float, help='balancing coef for CE and KD')
parser.add_argument('--beta', default=4e-4, type=float, help='balancing factor')

args = parser.parse_args()
Criterion_CE = nn.CrossEntropyLoss()
Criterion_KD = KLLoss(args.temperature)
Criterion_MSE = nn.MSELoss()


def Base_train(epoch, model, optimizer, train_loader):
    print('\nEPOCH: %d' % epoch)
    train_loss = AverageMeter('Loss', ':.4e')
    train_top1 = AverageMeter('Acc@1', ':6.2f')
    train_top5 = AverageMeter('Acc@5', ':6.2f')
    train_start_time = time.time()
    model.train()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        emb, output = model(inputs)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        train_loss.update(loss.item(), inputs.size(0))
        train_top1.update(acc1[0], inputs.size(0))
        train_top5.update(acc5[0], inputs.size(0))

    print('Train ** \t Time Taken: %.2f sec' % (time.time() - train_start_time))
    print('Loss: %.3f | Acc@1: %.3f%% | Acc@5: %.3f%% |' % (train_loss.avg, train_top1.avg, train_top5.avg))


def Base_val(model, test_loader):
    val_loss = AverageMeter('loss', ':.4e')
    val_top1 = AverageMeter('Acc@1', ':6.2f')
    val_top5 = AverageMeter('Acc@5', ':6.2f')
    infer_time = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            emb, output = model(inputs)
            # output = torch.log_softmax(output, dim=-1)
            loss = F.cross_entropy(output, targets)
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            val_loss.update(loss.item(), inputs.size(0))
            val_top1.update(acc1[0], inputs.size(0))
            val_top5.update(acc5[0], inputs.size(0))
        print('Test ** \t Time Taken: %.2f sec' % (time.time() - infer_time))
        print('Loss: %.3f | Acc@1: %.3f%% | Acc@5: %.3f%% |' % (val_loss.avg, val_top1.avg, val_top5.avg))
    return val_top1.avg, val_top5.avg


if __name__ == '__main__':
    print(args)
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    train_loader, test_loader, num_class = load_cifar(args)

    # model = resnext50_32x4d(num_class=num_class, aux=None).to(DEVICE)
    model = Res_50(num_classes=num_class, aux=None).to(DEVICE)
    # model = ShuffleNetV2(num_classes=num_class, aux=None).to(DEVICE)
    # model = VGG19_bn(num_classes=num_class, aux=None).to(DEVICE)
    # model = MobileNetV2(num_classes=num_class, aux=None).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)  # 注意，不能使用 nesterov
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    scheduler.get_last_lr()

    time_log = datetime.now().strftime('%m_%d_%H-%M')
    folder_name = '{}_{}'.format(args.model_name, time_log)
    path = os.path.join(args.save_folder, folder_name)
    path = path.replace('\\', '/')

    if not os.path.exists('logs/' + path):
        os.makedirs('logs/' + path)
    with open('logs/{}/parameters.txt'.format(path), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    max_acc1 = 0
    max_acc5 = 0
    max_epoch = 0

    for epoch in range(1, args.epochs + 1):
        f = open(os.path.join("logs/" + path, 'log.txt'), "a")
        Base_train(epoch, model, optimizer, train_loader)

        acc1, acc5 = Base_val(model, test_loader)
        scheduler.step()

        if max_acc1 < acc1:
            max_acc1 = acc1
            max_acc5 = acc5
            max_epoch = epoch

            # os.system('rm ckpt/{}/accMax*'.format(path))
            # save_checkpoint(
            #     {
            #         'epoch': epoch,
            #         'state_dict': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'scheduler': scheduler.state_dict()
            #     },
            #     True, 'ckpt/' + path, filename='accMax_{}.pth'.format(epoch))

        f.write('EPOCH: {epoch} \t ACC@1: {acc_1:.4f} \t ACC@5: {acc_5:.4f} \t'
                'BEST: {bestep} \t Max@1: {max_1:.4f} \t Max@5: {max_5:.4f} \n'.format(
            epoch=epoch, acc_1=acc1, acc_5=acc5,
            bestep=max_epoch, max_1=max_acc1,
            max_5=max_acc5)
        )
        print('BEST *** acc@1: {:.3f}%   *** acc@5: {:.3f}%  *** epoch:{} '.format(max_acc1, max_acc5, max_epoch))
        f.close()

        gc.collect()

