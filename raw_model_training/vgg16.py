import torch
from torch import nn
import numpy as np
from torch.autograd import Variable

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer


class VGG16_or(nn.Module):
    def __init__(self, dataset='128'):
        super(VGG16_or, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([1,64], [64,64], [3,3], [3,3], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [3,3], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        if dataset == '128':
            num_classes = 11
            self.layer5 = vgg_fc_layer(4608, 256)
        elif dataset == '512':
            num_classes = 12
            self.layer5 = vgg_fc_layer(16896, 256)
        elif dataset == '1024':
            num_classes = 24
            self.layer5 = vgg_fc_layer(33280, 256)
        elif dataset == '3040':
            num_classes = 106
            self.layer5 = vgg_fc_layer(97792, 256)
        self.layer6 = vgg_fc_layer(256, 128)

        # Final layer
        self.layer7 = nn.Linear(128, num_classes)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(-1, self.num_flat_feature(out))
        # print(out.shape)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return out
    
    def num_flat_feature(self,x):
        size=x.size()[1:]
        num_feature=1
        for s in size:
            num_feature*=s
        # print("num_feature",num_feature)
        return num_feature





class VGG16(nn.Module):
    def __init__(self, n_classes=24):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([1,64], [64,64], [3,3], [3,3], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [3,3], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(512, 256)
        self.layer7 = vgg_fc_layer(256, 128)

        # Final layer
        self.layer8 = nn.Linear(128, n_classes)

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(-1, self.num_flat_feature(vgg16_features))
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out


# from torchinfo import summary
# model = VGG16_or(dataset='3040').cuda()
# summary(model, input_size=(128, 1, 2, 3040))