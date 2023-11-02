from __future__ import print_function
import sys
import os

import numpy as np

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
import logging
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from dataset import  get_alldb_signal_train_validate_loader
from models.network import define_model
from models.resnet import *
from trades import trades_loss

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--dataset', type=str, default='128',
                    help='dataset using default 128')
parser.add_argument('--model', type=str, default='r8conv1',
                    help='models in exp default = CNN1D')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train, ori_default=76')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate 0.1')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
parser.add_argument('--epsilon', default=0.06,
                    help='perturbation')
parser.add_argument('--num-steps', default=5,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.03,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES default 1')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='../DefenseEnhancedModels/TRADES',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

args = parser.parse_args()

# settings
model_dir = args.model_dir +"{}".format(args.model)+ "{}".format(args.dataset)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
# ])
# trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
# testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def get_signal_train_validate_loader(batch_size, shuffle=True, random_seed=100, num_workers=1):
    """

    :param dir_name:
    :param batch_size:
    :param valid_size:
    :param augment:
    :param shuffle:
    :param random_seed:
    :param num_workers:
    :return:
    """
    train_2 = np.load('/home/zjut/public/data0/000_Dataset/001_Signal/dataset/radio{}NormTrainX.npy'.format(args.dataset))
    train_label_path = '/home/zjut/public/data0/000_Dataset/001_Signal/dataset/radio{}NormTrainSnrY.npy'.format(args.dataset)  # 训练集标签
    test_2 = np.load('/home/zjut/public/data0/000_Dataset/001_Signal/dataset/radio{}NormTestX.npy'.format(args.dataset))
    test_label_path = '/home/zjut/public/data0/000_Dataset/001_Signal/dataset/radio{}NormTestSnrY.npy'.format(args.dataset)  # 测试集标签

    if args.dataset == '3040':
        train_label = np.load(train_label_path)  # 得到0到11的类标签数据
        test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    else:
        train_label = np.load(train_label_path)[:, 0]  # 得到0到11的类标签数据
        test_label = np.load(test_label_path)[:, 0]  # 得到0到11的类标签数据

    # 数组变张量
    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'Vgg16t' or args.model=='r8conv1':
        print('Data will be reshaped into [N, 1, 2, Length]')
        train_2 = np.reshape(train_2, (train_2.shape[0], 1, train_2.shape[1], 2))
        test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
        # 数组变张量
        train_2 = torch.from_numpy(train_2).permute(0, 1, 3, 2)  # [312000, 1, 2, 128]
        train_2 = train_2.type(torch.FloatTensor)
        test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)
    else:
        train_2 = torch.from_numpy(train_2).permute(0, 2, 1)  # [312000, 2, 128] 统一转为[N, Channel, Length]形式
        train_2 = train_2.type(torch.FloatTensor)
        test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label)
    train_label = train_label.type(torch.LongTensor)
    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.LongTensor)
    print(train_2.shape, train_label.shape, test_2.shape, test_label.shape)
    # 把数据放在数据库中
    train_signal = torch.utils.data.TensorDataset(train_2, train_label)
    test_signal = torch.utils.data.TensorDataset(test_2, test_label)

    train_loader =torch.utils.data.DataLoader(dataset=train_signal, batch_size=batch_size,
                                                         shuffle=True, num_workers=num_workers, drop_last=True)
    validate_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_workers, drop_last=True)

    # load the dataset
    return train_loader, validate_loader


# if args.dataset == '128' or '512' or '1024' or '3040':
if args.dataset == '128':
    train_loader, test_loader = get_alldb_signal_train_validate_loader(batch_size=args.batch_size, shuffle=True)
elif args.dataset == '1024':
    train_loader, test_loader = get_signal_train_validate_loader(batch_size=args.batch_size, shuffle=True)
else:
    print("data error")


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    scale   = 0.1
    lr_list =  [args.lr] * 100
    lr_list += [args.lr*scale] * 50
    lr_list += [args.lr*scale*scale] * 50
    lr = lr_list[epoch]
    logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print('The **learning rate** of the {} epoch is {}'.format(epoch, param_group['lr']))

    # lr = args.lr
    # if epoch >= 75:
    #     lr = args.lr * 0.1
    # if epoch >= 90:
    #     lr = args.lr * 0.01
    # if epoch >= 100:
    #     lr = args.lr * 0.001
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
    model = define_model(name=args.model).to(device)
    # 对抗训练中用的是Adam
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    print(args.model + "is in trades!")

    for epoch in range(0, args.epochs):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, '{}-epoch{}.pt'.format(args.model, epoch)))
            # torch.save(optimizer.state_dict(),
            #            os.path.join(model_dir, 'opt--checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()