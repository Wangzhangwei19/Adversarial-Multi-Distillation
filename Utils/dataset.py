import numpy as np
import torch.utils.data as Data
import torch
from args import args

# help functions to get the training/validation/testing data loader

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
    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'Vgg16t' :
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

def get_signal_test_loader( batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """

    # test_2 = np.load('/home/zjut/public/data0/000_Dataset/001_Signal/dataset/radio{}NormTestX.npy'.format(args.dataset))
    # test_label_path = '/home/zjut/public/data0/000_Dataset/001_Signal/dataset/radio{}NormTestSnrY.npy'.format(args.dataset)  # 训练集标签
    # test_2 = np.load('/home/zjut/public/signal/wzw/SignalAttack/KD/CleanDatasets/target/dataT4.npy')
    # test_label_path = '/home/zjut/public/signal/wzw/SignalAttack/KD/CleanDatasets/target/labelT4.npy'  # 训练集标签
    # print('TSNE USE')
    # test_2 = np.load('/public/mount_data/data/wzw/dataset/radio128NormTestX.npy')
    # test_label_path = '/public/mount_data/data/wzw/dataset/radio128NormTestSnrY.npy'
    test_2 = np.load('/public/wzw/data/512_upper10db/radio512NormTestX.npy')
    test_label_path = '/public/wzw/data/512_upper10db/radio512NormTestSnrY.npy'

    if args.dataset == '3040':
        test_label = np.load(test_label_path)  # 得到0到11的类标签数据[
    else:
        test_label = np.load(test_label_path)[:, 0]  # 得到0到11的类标签数据

    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet':
        print('Data will be reshaped into [N, 1, 2, Length]')
        test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
        # 数组变张量
        test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)
    else:
        test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)

    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.LongTensor)
    print(test_2.shape, test_label.shape)

    # test_I = np.load(test_path_I)  # [44000, 128]
    # test_Q = np.load(test_path_Q)  # [44000, 128]
    # test_2 = np.stack([test_I, test_Q], axis=-1)  # [44000, 128, 2]
    # test_label = np.load(test_label_path)[1]  # 得到0到11的类标签数据
    # test_2 = torch.from_numpy(test_2).permute(0, 2, 1) # [156000, 2, 128]数组转tensor，021转换索引
    # test_label = torch.from_numpy(test_label)
    # # print(test_2.dtype,"1111111111111111",test_label.dtype)
    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_worker, drop_last=True)

    return test_loader


def get_alldb_signal_train_validate_loader(batch_size, shuffle=True, random_seed=100, num_workers=1):
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
    train_2 = np.load('/public/wzw/Series_exp/filterData/radio11CNormTrainX.npy')
    train_label_path = '/public/wzw/Series_exp/filterData/radio11CNormTrainSnrY.npy'  # 训练集标签
    test_2 = np.load('/public/wzw/Series_exp/filterData/radio11CNormTestX.npy')
    test_label_path = '/public/wzw/Series_exp/filterData/radio11CNormTestSnrY.npy'  # 测试集标签

    if args.dataset == '3040':
        train_label = np.load(train_label_path)  # 得到0到11的类标签数据
        test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    else:
        train_label = np.transpose(np.load(train_label_path),(1,0))[:, 1]
        test_label = np.transpose(np.load(test_label_path), (1, 0))[:, 1]
        # train_label = np.load(train_label_path)[:, 0]  # 得到0到11的类标签数据
        # test_label = np.load(test_label_path)[:, 0]  # 得到0到11的类标签数据

    # 数组变张量
    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'resnet8':
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

def get_alldb_signal_test_loader( batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """
    #
    # test_2 = np.load('/public/wzw/Series_exp/filterData/128_High_radio11CNormTestX.npy')
    # test_2 = np.load('/public/wzw/Series_exp/filterData/128_High_radio11CNormTestX.npy')
    test_2 = np.load('/public/wzw/Series_exp/filterData/radio11CNormTestX.npy')
    test_label_path = '/public/wzw/Series_exp/filterData/radio11CNormTestSnrY.npy'  # 训练集标签

    if args.dataset == '3040':
        test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    else:
        test_label = np.transpose(np.load(test_label_path), (1,0))[:, 1]
         # 得到0到11的类标签数据

    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'resnet8':
        print('Data will be reshaped into [N, 1, 2, Length]')
        test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
        # 数组变张量
        test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)
    else:
        test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)

    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.LongTensor)
    print(test_2.shape, test_label.shape)

    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_worker, drop_last=True)

    return test_loader

def ttttt_test_loader( batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """
    #
    test_2 = np.load('/home/zjut/public/signal/wzw/SignalAttack/KD/CleanDatasets/CNN1D/128/128_inputs.npy')
    test_label_path = '/home/zjut/public/signal/wzw/SignalAttack/KD/CleanDatasets/CNN1D/128/128_labels.npy'  # 训练集标签
    # test_2 = np.load('/home/wzw/data/128a_singledb_nor/{}db_NormTestX.npy'.format(args.db))
    # test_label_path = '/home/wzw/data/128a_singledb_nor/{}db_NormTestSnrY.npy'.format(args.db)  # 训练集标签

    if args.dataset == '3040':
        test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    else:
        test_label = np.transpose(np.load(test_label_path), (1,0))[:, 1]
         # 得到0到11的类标签数据

    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'resnet8':
        print('Data will be reshaped into [N, 1, 2, Length]')
        test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
        # 数组变张量
        test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)
    else:
        test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)

    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.LongTensor)
    print(test_2.shape, test_label.shape)

    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_worker, drop_last=True)

    return test_loader

def get_single_db_signal_test_loader( batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """
    #
    # test_2 = np.load('/home/wzw/data/128a-all-nor/radio11CNormTestX.npy')
    # test_label_path = '/home/wzw/data/128a-all-nor/radio11CNormTestSnrY.npy'  # 训练集标签

    test_2 = np.load('/public/wzw/data/128a_singledb_nor/{}db_NormTestX.npy'.format(args.db))
    test_label_path = '/public/wzw/data/128a_singledb_nor/{}db_NormTestSnrY.npy'.format(args.db)  # 训练集标签

    print('chosen data db is {}'.format(args.db))

    if args.dataset == '3040':
        test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    else:
        test_label = np.transpose(np.load(test_label_path), (1,0))[:, 1]
         # 得到0到11的类标签数据

    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'mobilenet' or args.model == 'resnet8':
        print('Data will be reshaped into [N, 1, 2, Length]')
        test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
        # 数组变张量
        test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)
    else:
        test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)

    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.LongTensor)
    print(test_2.shape, test_label.shape)

    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_worker, drop_last=False)

    return test_loader

def get_upper_minus4db_signal_test_loader( batch_size, shuffle=False, num_worker=1):
    """

    :param dir_name:
    :param batch_size:
    :param shuffle:
    :param num_worker:
    :return:
    """
    #
    # test_2 = np.load('/home/wzw/data/128a-all-nor/radio11CNormTestX.npy')
    # test_label_path = '/home/wzw/data/128a-all-nor/radio11CNormTestSnrY.npy'  # 训练集标签

    test_2 = np.load('/public/wzw/data/128upper-4db/TestX.npy')
    test_label_path = '/public/wzw/data/128upper-4db/TestY.npy' # 训练集标签
    print('Using upper minus 4 db signal dataset to test\n')

    if args.dataset == '3040':
        test_label = np.load(test_label_path)  # 得到0到11的类标签数据
    else:
        test_label = np.transpose(np.load(test_label_path), (1,0))[:, 1]
         # 得到0到11的类标签数据

    if args.model == 'Lenet' or args.model == 'Vgg16' or args.model == 'Alexnet' or args.model == 'mobilenet' or args.model == 'resnet8':
        print('Data will be reshaped into [N, 1, 2, Length]')
        test_2 = np.reshape(test_2, (test_2.shape[0], 1, test_2.shape[1], 2))
        # 数组变张量
        test_2 = torch.from_numpy(test_2).permute(0, 1, 3, 2)  # [156000, 1, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)
    else:
        test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [156000, 2, 128]
        test_2 = test_2.type(torch.FloatTensor)

    test_label = torch.from_numpy(test_label)
    test_label = test_label.type(torch.LongTensor)
    print(test_2.shape, test_label.shape)

    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_signal, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=num_worker, drop_last=False)

    return test_loader