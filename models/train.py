"""
CNN1D
"""
import argparse
import os
import numpy as np
import time
import csv
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import CNN1D, CNN2D, lstm, gru, mcldnn, Alexnet, LeNet, vgg16
from torch.autograd import Variable
import torch.nn.functional as F
import pickle


def train_model(args, batch_signal, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=50):
    since = time.time()  # 开始训练的时间
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            signal_num = 0
            # Iterate over data.
            pbar = tqdm(batch_signal[phase])
            for inputs, labels in pbar:
                inputs = inputs.cuda()  # (batch_size, 2, 128)
                labels = labels.cuda()  # (batch_size, )
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                signal_num += inputs.size(0)
                # 在进度条的右边实时显示数据集类型、loss值和精度
                epoch_loss = running_loss / signal_num
                epoch_acc = running_corrects.double() / signal_num
                pbar.set_postfix({'Set': '{}'.format(phase),
                                  'Loss': '{:.4f}'.format(epoch_loss),
                                  'Acc': '{:.4f}'.format(epoch_acc)})
                # print('\r{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), end=' ')
            if phase == 'train':
                scheduler.step()
            # 显示该轮的loss和精度
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            # 保存当前的训练集精度、测试集精度和最高测试集精度
            with open("../result/result_{}_{}.csv".format(args.dataset, args.model), 'a', newline='') as t1:
                writer_train1 = csv.writer(t1)
                writer_train1.writerow([epoch, phase, epoch_loss, epoch_acc, best_acc])
        # 保存测试精度最高时的模型参数
        torch.save(best_model_wts, "../result/model/{}_{}_best_lr={}.pth".format(args.dataset, args.model, args.lr))
        print('Best test Acc: {:4f}'.format(best_acc))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def prepare_data(args):
    # 导入数据集
    # data_path_I = '/home/qkf/qiukunfeng/data/signal_npy/my_11_original_train_I.npy'  # I训练集
    # data_path_Q = '/home/qkf/qiukunfeng/data/signal_npy/my_11_original_train_Q.npy'  # Q训练集
    train_2 = np.load('/public/MountData/DataDir/zjc/Project_attack/dataset/radio{}NormTrainX.npy'.format(args.dataset))
    train_label_path = '/public/MountData/DataDir/zjc/Project_attack/dataset/radio{}NormTrainSnrY.npy'.format(args.dataset)  # 训练集标签
    # test_path_I = '/home/qkf/qiukunfeng/data/signal_npy/my_11_original_test_I.npy'  # I测试集
    # test_path_Q = '/home/qkf/qiukunfeng/data/signal_npy/my_11_original_test_Q.npy'  # Q测试集
    test_2 = np.load('/public/MountData/DataDir/zjc/Project_attack/dataset/radio{}NormTestX.npy'.format(args.dataset))
    test_label_path = '/public/MountData/DataDir/zjc/Project_attack/dataset/radio{}NormTestSnrY.npy'.format(args.dataset)  # 训练集标签
    # test_label_path = '/home/qkf/qiukunfeng/data/signal_npy/radio11CNormTestSnrY.npy'  # 测试集标签
    # train_I = np.load(data_path_I)  # [176000, 128]
    # train_Q = np.load(data_path_Q)  # [176000, 128]
    # train_2 = np.stack([train_I, train_Q], axis=-1)  # [176000, 128, 2]
    # test_I = np.load(test_path_I)  # [44000, 128]
    # test_Q = np.load(test_path_Q)  # [44000, 128]
    # test_2 = np.stack([test_I, test_Q], axis=-1)  # [44000, 128, 2]
    if args.dataset == '3040':
        train_label = np.load(train_label_path)  # 得到0到11的类标签数据
        test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    else:
        train_label = np.load(train_label_path)[:, 0]  # 得到0到11的类标签数据
        test_label = np.load(test_label_path)[:, 0]  # 得到0到11的类标签数据
    # 训练集和测试集的数量
    # print(train_label.shape)
    # print(train_label[0])
    # print(test_label.shape)
    # print(test_label[0])
    data_sizes = {'train': len(train_label), 'test': len(test_label)}
    # 数组变张量
    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet':
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
    # 将训练集和测试集分批
    batch_signal = {'train': torch.utils.data.DataLoader(dataset=train_signal, batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.num_workers),
                    'test': torch.utils.data.DataLoader(dataset=test_signal, batch_size=args.batch_size,
                                                        shuffle=False, num_workers=args.num_workers)}
    return batch_signal, data_sizes


def arg_parse():
    parser = argparse.ArgumentParser(description='Signal_diagonal_matrix_CNN arguments.')
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int, help='Number of workers to load data.')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Dataset: 128, 512, 1024, 3040')
    parser.add_argument('--model', dest='model', type=str, help='Model: CNN1D, CNN2D, LSTM, GRU, MCLDNN, Lenet, Vgg16, Alexnet')
    parser.set_defaults(lr=0.001, batch_size=128, num_epochs=50, num_workers=8, dataset='128', model='CNN1D')
    return parser.parse_args()


def main():
    prog_args = arg_parse()
    with open("../result/result_{}_{}.csv".format(prog_args.dataset, prog_args.model), 'a', newline='') as t:
        writer_train = csv.writer(t)
        writer_train.writerow(['dataset={}, model={}, num_epoch={}, lr={}, batch_size={}'.format(prog_args.dataset, prog_args.model, prog_args.num_epochs, prog_args.lr, prog_args.batch_size)])
        writer_train.writerow(['epoch', 'phase', 'epoch_loss', 'epoch_acc', 'best_acc'])
    batch_signal, data_sizes = prepare_data(prog_args)  # 跳转到prepare_data函数，得到批训练集和批测试集
    # 模型放到GPU上
    if prog_args.model == 'CNN1D':
        model_ft = CNN1D.ResNet1D(prog_args.dataset).cuda()
    elif prog_args.model == 'CNN2D':
        model_ft = CNN2D.CNN2D(prog_args.dataset).cuda()
    elif prog_args.model == 'LSTM':
        model_ft = lstm.lstm2(prog_args.dataset).cuda()
    elif prog_args.model == 'GRU':
        model_ft = gru.gru2(prog_args.dataset).cuda()
    elif prog_args.model == 'MCLDNN':
        model_ft = mcldnn.MCLDNN(prog_args.dataset).cuda()
    elif prog_args.model == 'Lenet':
        model_ft = LeNet.LeNet_or(prog_args.dataset).cuda()
    elif prog_args.model == 'Vgg16':
        model_ft = vgg16.VGG16_or(prog_args.dataset).cuda()
    elif prog_args.model == 'Alexnet':
        model_ft = Alexnet.AlexNet_or(prog_args.dataset).cuda()
    else:
        print('Error! There is no model of your choice')
    print(model_ft)
    # 导入预训练模型
    # model_ft.load_state_dict(torch.load(r'D:\program_Q\new\torchvision_model\resnet50-19c8e357.pth'))
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    # optimizer_ft = optim.SGD([{"params": model_ft.parameters()}, {"params": filter_all}, {"params": bias_all}],
    #                          lr=prog_args.lr, momentum=0.9)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=prog_args.lr)
    # 学习率衰减
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.8)
    # 训练模型
    train_model(prog_args, batch_signal, data_sizes, model_ft, criterion, optimizer_ft,
                exp_lr_scheduler, num_epochs=prog_args.num_epochs)


if __name__ == '__main__':
    main()
