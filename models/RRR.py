# '''
# Properly implemented ResNet-s for CIFAR10 as described in paper [1].
# The implementation and structure of this file is hugely influenced by [2]
# which is implemented for ImageNet and doesn't have option A for identity.
# Moreover, most of the implementations on the web is copy-paste from
# torchvision's resnet and has wrong number of params.
# Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
# number of layers and parameters:
# name      | layers | params
# ResNet20  |    20  | 0.27M
# ResNet32  |    32  | 0.46M
# ResNet44  |    44  | 0.66M
# ResNet56  |    56  | 0.85M
# ResNet110 |   110  |  1.7M
# ResNet1202|  1202  | 19.4m
# which this implementation indeed has.
# Reference:
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Deep Residual Learning for Image Recognition. arXiv:1512.03385
# [2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# If you use this implementation in you work, please don't forget to mention the
# author, Yerlan Idelbayev.
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as init
#
# from torch.autograd import Variable
#
# __all__ = ['ResNet','resnet8', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
#
# def _weights_init(m):
#     classname = m.__class__.__name__
#     #print(classname)
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
#         init.kaiming_normal_(m.weight)
#
# class LambdaLayer(nn.Module):
#     def __init__(self, lambd):
#         super(LambdaLayer, self).__init__()
#         self.lambd = lambd
#
#     def forward(self, x):
#         return self.lambd(x)
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1, option='A'):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             if option == 'A':
#                 """
#                 For CIFAR10 ResNet paper uses option A.
#                 """
#                 self.shortcut = LambdaLayer(lambda x:
#                                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
#             elif option == 'B':
#                 self.shortcut = nn.Sequential(
#                      nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                      nn.BatchNorm1d(self.expansion * planes)
#                 )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=11):
#         super(ResNet, self).__init__()
#         self.in_planes = 16
#
#         self.conv1 = nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.linear = nn.Linear(64, num_classes)
#
#         self.apply(_weights_init)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         print(x.shape)
#         out = F.relu(self.bn1(self.conv1(x)))
#         print(out.shape)
#         out = self.layer1(out)
#         print(out.shape)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool1d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
#
# def resnet8():
#     return ResNet(BasicBlock, [1, 1, 1])
#
#
# def resnet20():
#     return ResNet(BasicBlock, [3, 3, 3])
#
#
# def resnet32():
#     return ResNet(BasicBlock, [5, 5, 5])
#
#
# def resnet44():
#     return ResNet(BasicBlock, [7, 7, 7])
#
#
# def resnet56():
#     return ResNet(BasicBlock, [9, 9, 9])
#
#
# def resnet110():
#     return ResNet(BasicBlock, [18, 18, 18])
#
#
# def resnet1202():
#     return ResNet(BasicBlock, [200, 200, 200])
#
#
# def test(net):
#     import numpy as np
#     total_params = 0
#
#     for x in filter(lambda p: p.requires_grad, net.parameters()):
#         total_params += np.prod(x.data.numpy().shape)
#     print("Total number of params", total_params)
#     print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))
#
#
# # if __name__ == "__main__":
# #     for net_name in __all__:
# #         if net_name.startswith('resnet'):
# #             print(net_name)
# #             test(globals()[net_name]())
# #             print()
#
# data = torch.randn(10,2,128)
# model = resnet8()
# out = model(data)
# print(out.shape)




from __future__ import absolute_import
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
import os
from args import args
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        # cfg should be a number in this case
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = nn.BatchNorm1d(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def downsample_basic_block(x, planes):
    x = F.adaptive_avg_pool1d(x, 1)
    #x = nn.AvgPool2d(2,2)(x)
    zero_pads = torch.Tensor(
        x.size(0), planes - x.size(1),  x.size(2)).zero_()
    if isinstance(x.data, torch.cuda.FloatTensor):
        # zero_pads = zero_pads.cuda()
        zero_pads = zero_pads.to(device)

    out = Variable(torch.cat([x.data, zero_pads], dim=1))

    return out

class ResNet(nn.Module):

    def __init__(self, depth, dataset='cifar10', cfg=None):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BasicBlock
        if cfg == None:
            cfg = [[16]*n, [32]*n, [64]*n]
            cfg = [item for sub_list in cfg for item in sub_list]

        self.cfg = cfg

        self.inplanes = 16
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,bias=False)
        self.conv1 = nn.Conv1d(2, 16, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[n:2*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[2*n:3*n], stride=2)
        self.avgpool = nn.AvgPool1d(8)
        if args.dataset == '1024':
            num_classes = 24 ####################
        elif args.dataset == '128':
            num_classes = 11
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # print(m.kernel_size)
                n = m.kernel_size[0] *  m.out_channels#m.kernel_size[1] *
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes*block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = F.adaptive_avg_pool1d(x, 1)
        #x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        fea = x
        feats = {}
        feats["feats"] = fea

        x = self.fc(x)

        return x#, feats

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

# if __name__ == '__main__':
#     net = resnet(depth=56)
#     x=Variable(torch.FloatTensor(16, 3, 32, 32))
#     y = net(x)
#     print(y.data.shape)

# data = torch.randn(10,2,128)
# model = resnet(depth=8)
# out = model(data)
# print(out.shape)