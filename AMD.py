from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst
import tqdm
from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from models.network import define_model
from kd_losses import *
from Utils.dataset import get_signal_test_loader, get_signal_train_validate_loader
from Utils.dataset import get_alldb_signal_test_loader, get_alldb_signal_train_validate_loader
from args import args


args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

global device
# device = 'cuda:%d' % args.gpu_index
device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')

cudnn.benchmark = True


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    # logging.info("args = %s", args)
    # logging.info("unparsed_args = %s", unparsed)

    logging.info('----------- Network Initialization --------------')
    snet = define_model(name=args.s_name)
    checkpoint = torch.load(args.s_init, map_location='cuda:{}'.format(args.gpu_index))

    load_pretrained_model(snet, checkpoint['net'])
    logging.info('Student param size = %fMB', count_parameters_in_MB(snet))

    # multi teacher setting
    # ACC Teacher
    t1net = define_model(name=args.t1_name)
    t1net.load_state_dict(torch.load(args.t1_model, map_location='cuda:{}'.format(args.gpu_index)))  # ["net"]

    t1net.eval()
    for param in t1net.parameters():
        param.requires_grad = False
    logging.info('Teacher1 param size = %fMB', count_parameters_in_MB(t1net))
    logging.info('-----------------------------------------------')

    # Adv Teacher
    t2net = define_model(name=args.t2_name)
    checkpoint = torch.load(args.t2_model, map_location='cuda:{}'.format(args.gpu_index))
    load_pretrained_model(t2net, checkpoint['net'])
    t2net.eval()
    for param in t2net.parameters():
        param.requires_grad = False
    # logging.info('Teacher: %s', t1net)
    logging.info('Teacher2 param size = %fMB', count_parameters_in_MB(t2net))
    logging.info('-----------------------------------------------')

    # define loss functions
    if args.kd_mode == 'logits':
        criterionKD = Logits().cuda(device)
    else:
        raise Exception('Invalid kd mode...')

    criterionCls = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, snet.parameters()), lr=args.lr)

    train_loader, valid_loader = get_alldb_signal_train_validate_loader(batch_size=args.batch_size, shuffle=True)
    test_loader = get_alldb_signal_test_loader(batch_size=args.batch_size, shuffle=False)


    # warp nets and criterions for train and test
    nets = {'snet': snet, 'tnet1': t1net, 'tnet2': t2net}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    # first initilizing the student nets

    best_top1 = 0
    best_top5 = 0
    for epoch in range(1, args.epochs + 1):
        adjust_lr(optimizer, epoch)

        # train one epoch
        epoch_start_time = time.time()
        train(train_loader, nets, optimizer, criterions, epoch)

        # evaluate on testing set
        logging.info('Testing the models......')
        test_top1, test_top5 = test(test_loader, nets, criterions, epoch)

        epoch_duration = time.time() - epoch_start_time
        logging.info('Epoch time: {}s'.format(int(epoch_duration)))

        # save model
        is_best = False
        if test_top1 > best_top1:
            best_top1 = test_top1
            best_top5 = test_top5
            is_best = True
            # if is_best :
            logging.info('Saving models......')
        save_checkpoint({
            'epoch': epoch,
            'net': snet.state_dict(),
            'prec@1': test_top1,
            'prec@5': test_top5,
        }, is_best, args.save_root, epoch)


#  -adversarial training setting-

# step_size = 0.05
step_size = args.step_size
epsilon = args.epsilon


def train(train_loader, nets, optimizer, criterions, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cls_losses = AverageMeter()
    kd_losses1 = AverageMeter()
    kd_losses2 = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    tnet1 = nets['tnet1']
    tnet2 = nets['tnet2']

    criterionCls = criterions['criterionCls']
    criterionKD = criterions['criterionKD']

    snet.train()

    # train_loader = tqdm(train_loader)

    for i, (img, target) in enumerate(train_loader, start=1):
        # print(img.shape,'---------------')
        img = torch.squeeze(img, dim=1)

        img_numpy = img.cpu().numpy()
        copy_images = img_numpy.copy()
        img = img.cuda(device, non_blocking=True)
        target = target.cuda(device, non_blocking=True)

        # for para in snet.bn1.parameters():
        #    para.requires_grad = False

        copy_images = copy_images + np.random.uniform(-epsilon, epsilon, copy_images.shape).astype('float32')

        for j in range(args.step):  # num step
            # numpy.random.uniform(low,high,size)
            var_copy_images = torch.from_numpy(copy_images).to(device)
            var_copy_images.requires_grad = True
            # print(var_copy_images.shape,'------------')
            preds = snet(var_copy_images)
            loss_preds = F.cross_entropy(preds, target)
            gradient = torch.autograd.grad(loss_preds, var_copy_images)[0]
            gradient_sign = torch.sign(gradient).cpu().numpy()

            copy_images = copy_images + step_size * gradient_sign
            copy_images = np.clip(copy_images, img_numpy - epsilon, img_numpy + epsilon)
            copy_images = np.clip(copy_images, -1, 1)
        adv_img = torch.from_numpy(copy_images).to(device)

        out_s = snet(img)
        # img = torch.unsqueeze(img, dim=1)
        out_t1 = tnet1(img)
        cls_loss = criterionCls(out_s, target)
        adv_img = torch.unsqueeze(adv_img, dim=1)

        out_t2 = tnet2(adv_img)
        adv_img = torch.squeeze(adv_img, dim=1)
        out_s_a = snet(adv_img)
        cls_adv_loss = criterionCls(out_s_a, target)

        if args.kd_mode in ['logits', 'st']:
            kd_loss1 = criterionKD(out_s, out_t1.detach()) * args.lambda_kd1 * 0.1
            kd_loss2 = criterionKD(out_s_a, out_t2.detach()) * args.lambda_kd2 * 0.1

        else:
            raise Exception('Invalid kd mode...')

        # ---------------定义损失函数
        loss = cls_loss + kd_loss1 + kd_loss2 + cls_adv_loss * args.lambda_kd3

        prec1, prec5 = accuracy(out_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        kd_losses1.update(kd_loss1.item(), img.size(0))
        kd_losses2.update(kd_loss2.item(), img.size(0))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                       'Time:{batch_time.val:.4f} '
                       'Data:{data_time.val:.4f}  '
                       'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
                       'KD1:{kd_losses1.val:.4f}({kd_losses1.avg:.4f})  '
                       'KD2:{kd_losses2.val:.4f}({kd_losses2.avg:.4f})  '
                       'LOSS:{losses.val:.4f}({losses.avg:.4f})  '
                       'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                       'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                cls_losses=cls_losses, kd_losses1=kd_losses1, kd_losses2=kd_losses2,
                losses=losses, top1=top1, top5=top5))
            logging.info(log_str)


def test(test_loader, nets, criterions, epoch):
    cls_losses = AverageMeter()
    kd_losses1 = AverageMeter()
    kd_losses2 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    snet = nets['snet']
    tnet1 = nets['tnet1']
    tnet2 = nets['tnet2']

    criterionCls = criterions['criterionCls']
    criterionKD = criterions['criterionKD']

    snet.eval()
    end = time.time()
    for i, (img, target) in enumerate(test_loader, start=1):
        img = torch.squeeze(img, dim=1)
        img = img.cuda(device, non_blocking=True)
        target = target.cuda(device, non_blocking=True)

        if args.kd_mode in ['sobolev', 'lwm']:
            img.requires_grad = True
        else:
            with torch.no_grad():
                out_s = snet(img)

        cls_loss = criterionCls(out_s, target)

        prec1, prec5 = accuracy(out_s, target, topk=(1, 5))
        cls_losses.update(cls_loss.item(), img.size(0))
        # kd_losses1.update(kd_loss1.item(), img.size(0))
        # kd_losses2.update(kd_loss2.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [cls_losses.avg, top1.avg, top5.avg]
    logging.info('Cls: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))

    return top1.avg, top5.avg


def adjust_lr_init(optimizer, epoch):
    scale = 0.1
    lr_list = [args.lr * scale] * 30
    lr_list += [args.lr * scale * scale] * 10
    lr_list += [args.lr * scale * scale * scale] * 10

    lr = lr_list[epoch - 1]
    logging.info('Epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr(optimizer, epoch):
    scale = 0.1
    lr_list = [args.lr] * 100
    lr_list += [args.lr * scale] * 50
    lr_list += [args.lr * scale * scale] * 50

    lr = lr_list[epoch - 1]
    logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = False
    main()
