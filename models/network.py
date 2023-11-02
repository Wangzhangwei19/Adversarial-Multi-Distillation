import torch
import os
from args import args
from models.lstm import lstm2
from models.mcldnn import MCLDNN
from models.Alexnet import AlexNet_or
from models.CNN1D import ResNet1D
from models.CNN2D import CNN2D
from models.gru import gru2
from models.LeNet import LeNet_or
from models.vgg16 import VGG16_or
from models.mobilenet import mobilenet
from models.resnet import resnet
from models.RRR import resnet as r8
from models.vgg import vgg11_bn

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')

def define_model(name):
	if name == 'LSTM':
		net = lstm2(dataset=args.dataset).to(device)
	elif name == 'MCLDNN':
		net = MCLDNN(dataset=args.dataset).to(device)
	elif name == 'Alexnet':
		net = AlexNet_or(dataset=args.dataset).to(device)
	elif name == 'CNN1D':
		net = ResNet1D(dataset=args.dataset).to(device)
	elif name == 'CNN2D':
		net = CNN2D(dataset=args.dataset).to(device)
	elif name == 'Lenet':
		net = LeNet_or(dataset=args.dataset).to(device)
	elif name == 'Vgg16':
		net = VGG16_or(dataset=args.dataset).to(device)
	elif name == 'GRU':
		net = gru2(dataset=args.dataset).to(device)
	elif name == 'mobilenet':
		net = mobilenet().to(device)
	# elif name == 'resnet8':
	# 	net = resnet(depth=8).to(device)
	elif name == 'r8conv1':
		net = r8(depth=8).to(device)
	elif name == 'vgg11_bn':
		net = vgg11_bn().to(device)
	else:
		raise Exception('model name does not exist.')
	# if True:
	# 	# net = torch.nn.DataParallel(net).cuda()
	# 	net = net.to(device)
	# else:
	# 	net = torch.nn.DataParallel(net)

	return net