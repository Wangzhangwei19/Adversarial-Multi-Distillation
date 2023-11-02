import torch
import torch.nn as nn
class LeNet(nn.Module):
    def __init__(self,num_class=None):
        super(LeNet, self).__init__()
        self.conv1=nn.Conv2d(1,64,3,padding=2)
        self.bath1 = nn.BatchNorm2d(64)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(64,128,3,padding=2)
        self.bath2=nn.BatchNorm2d(128)
        self.pool2=nn.MaxPool2d(2,2)
        self.drop=nn.Dropout(p=0.5)
        # self.conv3 = nn.Conv2d(16, 32, 3)
        # self.bath3 = nn.BatchNorm2d(32)
        # self.pool3 = nn.MaxPool2d(2, 2)
        self.fc3=nn.Linear(512,120)
        self.fc4=nn.Linear(120,84)
        self.fc5=nn.Linear(84,num_class)

    def forward(self,x):
        # x = x.unsqueeze(dim=1)
        x=self.pool1(self.bath1(torch.relu(self.conv1(x))))
        x=self.pool2(self.bath2(torch.relu(self.conv2(x))))
        # x =self.pool3(self.bath3(torch.relu(self.conv3(x))))
        # print(x.size)
        x=x.view(-1,self.num_flat_feature(x))

        x=torch.relu(self.fc3(x))
        x = self.drop(x)
        x=torch.relu(self.fc4(x))
        x=self.fc5(x)
        return x
    def num_flat_feature(self,x):
        size=x.size()[1:]
        num_feature=1

        for s in size:
            num_feature*=s
        print("num_feature",num_feature)
        return num_feature

class LeNet_or(nn.Module):
    def __init__(self, dataset='128'):
        super(LeNet_or, self).__init__()
        self.conv1=nn.Conv2d(1,6,kernel_size=(2,3),padding=2)
        self.pool1=nn.MaxPool2d(2,2)
        self.drop=nn.Dropout(0.5)
        self.conv2=nn.Conv2d(6,16,kernel_size=(2,3),padding=2)
        self.pool2=nn.MaxPool2d(2,2)
        if dataset == '128':
            num_classes = 11
            self.fc_pool = nn.Linear(33, 33)  # 修改部分，否则128以上长度数据集训练效果较差
            self.fc3 = nn.Linear(1056, 500)
        elif dataset == '512':
            num_classes = 12
            self.fc_pool = nn.Linear(129, 128)  # 修改部分，否则128以上长度数据集训练效果较差
            self.fc3 = nn.Linear(4096, 500)
        elif dataset == '1024':
            num_classes = 24
            self.fc_pool = nn.Linear(257, 128)  # 修改部分，否则128以上长度数据集训练效果较差
            self.fc3 = nn.Linear(4096, 500)
        elif dataset == '3040':
            num_classes = 106
            self.fc_pool = nn.Linear(761, 128)  # 修改部分，否则128以上长度数据集训练效果较差
            self.fc3 = nn.Linear(4096, 500)
        self.fc4=nn.Linear(500,84)
        self.fc5=nn.Linear(84, num_classes)

    def forward(self,x):
        # x = x.unsqueeze(dim=1)
        x=self.pool1(torch.relu(self.conv1(x)))
        x=self.pool2(torch.relu(self.conv2(x)))
        # print(x.size)
        x=self.fc_pool(x)
        x=x.view(-1,self.num_flat_feature(x))
        # print(x.shape)
        x=torch.relu(self.fc3(x))
        x=self.drop(x)
        x=torch.relu(self.fc4(x))
        x=self.fc5(x)
        return x
    def num_flat_feature(self,x):
        size=x.size()[1:]
        num_feature=1
        for s in size:
            num_feature*=s
        # print("num_feature",num_feature)
        return num_feature

# from torchinfo import summary
# model = LeNet_or(dataset='1024').cuda()
# summary(model, input_size=(128, 1, 2, 1024))