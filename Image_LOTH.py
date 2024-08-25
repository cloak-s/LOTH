from utils import AverageMeter, accuracy, load_cifar, KLLoss, set_seed, sigmoid_rampup, load_tinyimagenet
import torch
import torch.nn as nn
from Backbone import Res_18, Res_34, Norm_fusion, MobileNetV2, VGG16_bn, Msa_Fusion, SCAN_Fusion, DECA_fusion, VGG19_bn
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
# parser.add_argument('--data-dir', default='E:\CNN\dataset\\tiny-imagenet-200', type=str, help='the diretory to save cifar100 dataset')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4 )')
parser.add_argument('--data-dir', default='./data/tiny-imagenet-200', type=str, help='the diretory to save cifar100 dataset')
parser.add_argument('--start-epoch', default=0, type=int, help='manual iter number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--epochs', default=200, type=int, help='number of total iterations (default:300)')
parser.add_argument('--milestones', default=[75, 130, 180], nargs="*", type=list)
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--model_name', default='VGG16-FFL', type=str)
parser.add_argument('--resume', default='', type=str, help='path to  latest checkpoint (default: None)')
parser.add_argument('--save-folder', default='LOTH-Tiny/', type=str, help='folder to save the checkpoints')
parser.add_argument('--cuda_num', default='0', type=str)
parser.add_argument('--temperature', default=10.0, type=int, help='temperature to smooth the logits')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--consistency_rampup', '--consistency_rampup', default=20, type=float,
                    metavar='consistency_rampup', help='consistency_rampup ratio')
parser.add_argument('--alpha', default=1.15, type=float, help='balancing coef for CE and KD')
parser.add_argument('--beta', default=4e-4, type=float, help='balancing factor')

args = parser.parse_args()
Criterion_CE = nn.CrossEntropyLoss()
Criterion_KD = KLLoss(args.temperature)
Criterion_MSE = nn.MSELoss()


def Train_LOTH(epoch, model, fuse_module, optimizer, optimizer_fm, train_loader):
    print('\nEPOCH: %d' % epoch)
    train_loss = AverageMeter('Train_Loss', ':.4e')
    acc_main = AverageMeter('acc@m', ':6.2f')
    Exit_1 = AverageMeter('Exit@1', ':6.2f')
    Exit_2 = AverageMeter('Exit@2', ':6.2f')
    Exit_3 = AverageMeter('Exit@3', ':6.2f')
    acc_f = AverageMeter('acc@f', ':6.2f')
    acc_e = AverageMeter('acc@e', ':6.2f')
    train_start_time = time.time()

    model.train()
    fuse_module.train()

    A = args.alpha
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        embs, logits = model(inputs)
        f_emb, f_logits = fuse_module(embs)
        ensemble_logit = sum(logits) / len(logits)

        optimizer.zero_grad()
        optimizer_fm.zero_grad()

        Total_loss = Criterion_CE(f_logits, targets) + Criterion_KD(f_logits, ensemble_logit)  # fusion_module
        for i, logit in enumerate(logits):
            Total_loss += (2 - A ** (4-i)) * Criterion_CE(logits[i], targets) \
                          + A ** (4-i) * Criterion_KD(logits[i], f_logits)

        # feature_loss = intra_fd(f_emb)
        # for emb in embs:
        #     feature_loss += intra_fd(emb)
        # Total_loss += 8e-4 * feature_loss

        Total_loss.backward()
        optimizer.step()
        optimizer_fm.step()

        E1, _ = accuracy(logits[0], targets, topk=(1, 5))
        E2, _ = accuracy(logits[1], targets, topk=(1, 5))
        E3, _ = accuracy(logits[2], targets, topk=(1, 5))
        Em, _ = accuracy(logits[-1], targets, topk=(1, 5))
        Ef, _ = accuracy(f_logits, targets, topk=(1, 5))
        Ee, _ = accuracy(ensemble_logit, targets, topk=(1, 5))

        train_loss.update(Total_loss.item(), inputs.size(0))
        Exit_1.update(E1[0], inputs.size(0))
        Exit_2.update(E2[0], inputs.size(0))
        Exit_3.update(E3[0], inputs.size(0))
        acc_main.update(Em[0], inputs.size(0))
        acc_f.update(Ef[0], inputs.size(0))
        acc_e.update(Ee[0], inputs.size(0))

    print('Train *** \t Time Taken: %.2f sec  \t  Loss: %.3f '  % (time.time() - train_start_time, train_loss.avg))
    print('Exit@1: %.3f%% | Exit@2: %.3f%% | Exit@3: %.3f%% | Exit@m: %.3f%% | Exit@f: %.3f%% | Exit@e: %.3f%% |'
          % (Exit_1.avg, Exit_2.avg, Exit_3.avg, acc_main.avg, acc_f.avg, acc_e.avg))


def Val_LOTH(model, fuse_module, test_loader):
    val_loss = AverageMeter('Val_Loss', ':.4e')
    Exit_1 = AverageMeter('Exit@1', ':6.2f')
    Exit_2 = AverageMeter('Exit@2', ':6.2f')
    Exit_3 = AverageMeter('Exit@3', ':6.2f')
    acc_f = AverageMeter('acc@f', ':6.2f')
    acc_e = AverageMeter('acc@e', ':6.2f')
    acc_main = AverageMeter('acc@m', ':6.2f')

    val_start_time = time.time()
    model.eval()
    fuse_module.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            embs, logits = model(inputs)
            _, f_logits = fuse_module(embs)
            ensemble_logit = sum(logits) / len(logits)

            loss = Criterion_CE(logits[-1], targets)

            E1, _ = accuracy(logits[0], targets, topk=(1, 5))
            E2, _ = accuracy(logits[1], targets, topk=(1, 5))
            E3, _ = accuracy(logits[2], targets, topk=(1, 5))
            Em, _ = accuracy(logits[-1], targets, topk=(1, 5))
            Ef, _ = accuracy(f_logits, targets, topk=(1, 5))
            Ee, _ = accuracy(ensemble_logit, targets, topk=(1, 5))

            val_loss.update(loss.item(), inputs.size(0))
            Exit_1.update(E1[0], inputs.size(0))
            Exit_2.update(E2[0], inputs.size(0))
            Exit_3.update(E3[0], inputs.size(0))
            acc_main.update(Em[0], inputs.size(0))
            acc_f.update(Ef[0], inputs.size(0))
            acc_e.update(Ee[0], inputs.size(0))

        print('Val *** \t Time Taken: %.2f sec  \t  Loss: %.3f ' % (time.time() - val_start_time, val_loss.avg))
        print('Exit@1: %.3f%% | Exit@2: %.3f%% | Exit@3: %.3f%% | Exit@m: %.3f%% | Exit@f: %.3f%% | Exit@e: %.3f%% |'
              % (Exit_1.avg, Exit_2.avg, Exit_3.avg, acc_main.avg, acc_f.avg, acc_e.avg))

        return acc_main.avg, acc_f.avg, acc_e.avg, Exit_1.avg, Exit_2.avg, Exit_3.avg


if __name__ == '__main__':
    print(args)
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    train_loader, test_loader, num_class = load_tinyimagenet(args)

    model = VGG16_bn(num_classes=num_class, aux='Bottle').to(DEVICE)
    # fuse_module = Norm_fusion(channel=512, layer=4, num_class=100).to(DEVICE)
    fuse_module = Norm_fusion(channel=512, num_class=num_class, layer=4).to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer_fm = optim.SGD(fuse_module.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    scheduler_fm = optim.lr_scheduler.MultiStepLR(optimizer_fm, milestones=args.milestones, gamma=0.1)

    time_log = datetime.now().strftime('%m_%d_%H-%M')
    folder_name = '{}_{}'.format(args.model_name, time_log)
    path = os.path.join(args.save_folder, folder_name)
    path = path.replace('\\', '/')
    # if not os.path.exists('ckpt/' + path):
    #     os.makedirs('ckpt/' + path)
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
        Train_LOTH(epoch, model, fuse_module, optimizer, optimizer_fm, train_loader)
        main_acc, f_acc, e_acc, exit_1, exit_2, exit_3 = Val_LOTH(model, fuse_module, test_loader)

        scheduler.step()
        scheduler_fm.step()

        if best_main_acc <= main_acc:
            best_main_acc = main_acc
            best_main_epo = epoch

        if best_f_acc <= f_acc:
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


