from utils import AverageMeter, accuracy, load_cifar, KLLoss, set_seed, sigmoid_rampup, load_tinyimagenet
import torch
import torch.nn as nn
from models import VGG16_bn, Res_34, VGG19_bn, MobileNetV1
from Backbone import Res_18
import argparse
import json
import time
from datetime import datetime
import warnings
import os
import gc
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda")
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 training')
# parser.add_argument('--data-dir', default='./data/CIFAR100', type=str, help='the diretory to save cifar100 dataset')
parser.add_argument('--data-dir', default='./data/tiny-imagenet-200', type=str, help='the diretory to save cifar100 dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4 )')
parser.add_argument('--start-epoch', default=0, type=int, help='manual iter number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--epochs', default=200, type=int, help='number of total iterations (default:300)')
parser.add_argument('--milestones', default=[75, 130, 180], nargs="*", type=list)
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--model_name', default='VGG19', type=str)
parser.add_argument('--resume', default='', type=str, help='path to  latest checkpoint (default: None)')
parser.add_argument('--save-folder', default='BEED_Tiny/', type=str, help='folder to save the checkpoints')
parser.add_argument('--cuda_num', default='0', type=str)
parser.add_argument('--temperature', default=3.0, type=int, help='temperature to smooth the logits')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--consistency_rampup', '--consistency_rampup', default=20, type=float,
                    metavar='consistency_rampup', help='consistency_rampup ratio')
parser.add_argument('--alpha', default= 1.15, type=float, help='balancing coef for CE and KD')
parser.add_argument('--beta', default=1.6, type=float, help='balancing factor')


args = parser.parse_args()
Criterion_CE = nn.CrossEntropyLoss()
Criterion_KD = KLLoss(args.temperature)
Criterion_MSE = nn.MSELoss()
writer = SummaryWriter('Result/Practice')


def Train_BEED(epoch, model, optimizer, train_loader):
    print('\nEPOCH: %d' % epoch)
    train_loss = AverageMeter('Loss', ':.4e')
    Acc1 = AverageMeter('Acc@1', ':6.2f')
    Acc2 = AverageMeter('Acc@2', ':6.2f')
    Acc3 = AverageMeter('Acc@3', ':6.2f')
    Acc4 = AverageMeter('Acc@4', ':6.2f')
    Acc_e = AverageMeter('acc@e', ':6.2f')
    train_start_time = time.time()
    model.train()
    A = args.alpha   # loss balancing coef
    B = args.beta    # importance coef for ensemble
    e_sum = args.beta ** 0 + args.beta ** 1 + args.beta ** 2 + args.beta ** 3
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        embs, logits = model(inputs)

        CE1 = Criterion_CE(logits[0], targets)
        CE2 = Criterion_CE(logits[1], targets)
        CE3 = Criterion_CE(logits[2], targets)
        CE4 = Criterion_CE(logits[3], targets)

        output_e = (
                B ** 0 * logits[0] / e_sum + B ** 1 * logits[1] / e_sum + B ** 2 * logits[2] / e_sum + B ** 3 * logits[-1] / e_sum).detach()
        feature_e = (
                B ** 0 * embs[0] / e_sum + B ** 1 * embs[1] / e_sum + B ** 2 * embs[2] / e_sum + B ** 3 * embs[-1] / e_sum).detach()

        KD1 = Criterion_KD(logits[0], output_e.detach())
        KD2 = Criterion_KD(logits[1], output_e.detach())
        KD3 = Criterion_KD(logits[2], output_e.detach())
        KD4 = Criterion_KD(logits[3], output_e.detach())

        FD1 = Criterion_MSE(embs[0], feature_e.detach())
        FD2 = Criterion_MSE(embs[1], feature_e.detach())
        FD3 = Criterion_MSE(embs[2], feature_e.detach())
        FD4 = Criterion_MSE(embs[3], feature_e.detach())

        total_loss = ((2 - A ** 3) * CE1 + A ** 3 * KD1) \
                     + ((2 - A ** 2) * CE2 + A ** 2 * KD2) \
                     + ((2 - A ** 1) * CE3 + A ** 1 * KD3) \
                     + ((2 - A ** 0) * CE4 + A ** 0 * KD4) \
                     + 0.1 * (FD1 + FD2 + FD3 + FD4)

        train_loss.update(total_loss.item(), inputs.size(0))
        prec1 = accuracy(logits[0].data, targets, topk=(1,))
        Acc1.update(prec1[0], inputs.size(0))
        prec2 = accuracy(logits[1].data, targets, topk=(1,))
        Acc2.update(prec2[0], inputs.size(0))
        prec3 = accuracy(logits[2].data, targets, topk=(1,))
        Acc3.update(prec3[0], inputs.size(0))
        prec4 = accuracy(logits[3].data, targets, topk=(1,))
        Acc4.update(prec4[0], inputs.size(0))
        Ee, _ = accuracy(output_e, targets, topk=(1, 5))
        Acc_e.update(Ee[0], inputs.size(0))

        total_loss.backward()
        optimizer.step()
    print('Training ** \t Time Taken: %.2f sec' % (time.time() - train_start_time))
    print('Loss: %.3f | Acc@1: %.3f%% | Acc@2: %.3f%% | Acc@3: %.3f%% | Acc@mian: %.3f%% | Exit@e: %.3f%% |' %
          (train_loss.avg, Acc1.avg, Acc2.avg,  Acc3.avg,  Acc4.avg, Acc_e.avg))


def Val_BEED(model, test_loader):
    val_loss = AverageMeter('Loss', ':.4e')
    Acc1 = AverageMeter('Acc@1', ':6.2f')
    Acc2 = AverageMeter('Acc@2', ':6.2f')
    Acc3 = AverageMeter('Acc@3', ':6.2f')
    Acc4 = AverageMeter('Acc@4', ':6.2f')
    Acce = AverageMeter('Acc@e', ':6.2f')
    val_start_time = time.time()
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        embs, logits = model(inputs)
        CE4 = Criterion_CE(logits[-1], targets)
        val_loss.update(CE4.item(), inputs.size(0))

        prec1, _ = accuracy(logits[0], targets, topk=(1,5))
        Acc1.update(prec1[0], inputs.size(0))
        prec2, _ = accuracy(logits[1], targets, topk=(1,5))
        Acc2.update(prec2[0], inputs.size(0))
        prec3, _ = accuracy(logits[2], targets, topk=(1,5))
        Acc3.update(prec3[0], inputs.size(0))
        prec4, _ = accuracy(logits[3], targets, topk=(1,5))
        Acc4.update(prec4[0], inputs.size(0))
        fused= sum(logits) / len(logits)
        # prece = accuracy((output1 + output2 + output3 + output4).data, target, topk=(1,))
        prece, _ = accuracy(fused, targets, topk=(1,5))
        Acce.update(prece[0], inputs.size(0))
    print('Training ** \t Time Taken: %.2f sec' % (time.time() - val_start_time))
    print('Loss: %.3f | Acc@1: %.3f%% | Acc@2: %.3f%% | Acc@3: %.3f%% | Acc@main: %.3f%% |Acc@e: %.3f%% |' %
          (val_loss.avg, Acc1.avg, Acc2.avg, Acc3.avg, Acc4.avg, Acce.avg))
    return Acc1.avg, Acc2.avg, Acc3.avg, Acc4.avg, Acce.avg


if __name__ == '__main__':
    print(args)
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    # train_loader, test_loader, num_class = load_cifar(args)
    train_loader, test_loader, num_class = load_tinyimagenet(args)

    model = VGG19_bn(num_classes=num_class, aux='Bottle').to(DEVICE)
    # model = Res_34(num_classes=num_class, aux='G-Bottle', before_relu=False).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
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

    best_main_acc = 0
    best_main_epo = 0
    best_f_acc = 0
    best_f_epo = 0

    for epoch in range(1, args.epochs + 1):
        f = open(os.path.join("logs/" + path, 'log.txt'), "a")
        Train_BEED(epoch, model, optimizer, train_loader)
        exit_1, exit_2, exit_3, main_acc, f_acc = Val_BEED(model, test_loader)

        scheduler.step()

        if best_main_acc < main_acc:
            best_main_acc = main_acc
            best_main_epo = epoch

        if best_f_acc < f_acc:
            best_f_acc = f_acc
            best_f_epo = epoch

        f.write('EPOCH: {EPOCH} \t Exit@1: {exit_1:.4f} \t Exit@2: {exit_2:.4f} \t Exit@3: {exit_3:.4f} \t'
                'Exit@main : {t_acc1:.4f} \t Max_main: {max_main:.4f} \t Epo_main: {epo_t} \t'
                'Exit@_Fusion : {f_acc1:.4f} \t Max_Fusion: {max_f:.4f} \n'.format(
            EPOCH=epoch, exit_1=exit_1, exit_2=exit_2, exit_3=exit_3,
            t_acc1=main_acc, max_main=best_main_acc, epo_t=best_main_epo,
            f_acc1=f_acc, max_f=best_f_acc)
        )
        print('Best Epoch: *** Epoch-T: {}  *** Epoch-F: {} '.format(best_main_epo, best_f_epo))
        print('Bset ACC:   *** Acc-T: %.3f%% *** best Acc-F: %.3f%% '
              % (best_main_acc, best_f_acc))

        f.close()
        gc.collect()
