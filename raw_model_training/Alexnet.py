import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
class AlexNet(nn.Module):
    def __init__(self, num_calss= None):
        super(AlexNet, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=(3,3),stride=1,padding=2),

            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,192,kernel_size=3,padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384,256,kernel_size=3,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.avgpool=nn.AdaptiveAvgPool2d((6,6))
        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,num_calss)
        )
    def forward(self,x):
        # x = x.unsqueeze(dim=1)
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x

class AlexNet_or(nn.Module):
    def __init__(self, dataset='128'):
        super(AlexNet_or, self).__init__()
        if dataset == '128':
            num_classes = 11
        elif dataset == '512':
            num_classes = 12
        elif dataset == '1024':
            num_classes = 24
        elif dataset == '3040':
            num_classes = 106
        self.features=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=(2,3),stride=2,padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,192,kernel_size=2,padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(192,384,kernel_size=2,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=2,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=2,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.avgpool=nn.AdaptiveAvgPool2d((6,6))
        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500,100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes)
        )
    def forward(self,x):
        # x = x.unsqueeze(dim=1)
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x

# from torchinfo import summary
# model = AlexNet_or(dataset='3040').cuda()
# summary(model, input_size=(128, 1, 2, 3040))



