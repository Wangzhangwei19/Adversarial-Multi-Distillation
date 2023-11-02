import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
# from RawModels.Utils.dataset import get_mnist_train_validate_loader, get_mnist_test_loader
# from RawModels.Utils.dataset import get_cifar10_train_validate_loader, get_cifar10_test_loader
from Utils.dataset import get_signal_train_validate_loader, get_signal_test_loader, get_alldb_signal_train_validate_loader
from models.network import define_model
from args import args

from Defenses.DefenseMethods.DD import DistillationDefense


def main():
    # Device configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the random seed manually for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Get training parameters, set up model frames and then get the train_loader and test_loader
    dataset = args.dataset.upper()
    if dataset == '128' or '512' or '1024' or '3040':
        model_framework = define_model(name=args.model).to(device)
        # raw train_loader (no augmentation) for constructing the SoftLabelDataset and then used to train the distilled model
        # 128 using all db dataset
        # train_loader, valid_loader = get_alldb_signal_train_validate_loader(batch_size=args.batch_size, shuffle=True)
        # 1024 using >=10db dataset
        train_loader, valid_loader = get_signal_train_validate_loader(batch_size=args.batch_size, shuffle=True)

        # testing dataset loader
        # test_loader = get_signal_test_loader(batch_size=args.batch_size, shuffle=False)
        test_loader = get_signal_test_loader(batch_size=args.batch_size, shuffle=False)

        # raw train_loader (no augmentation) for constructing the SoftLabelDataset and then used to train the distilled model
        # raw_train_loader, raw_valid_loader = get_alldb_signal_train_validate_loader(batch_size=args.batch_size, shuffle=True)
        raw_train_loader, raw_valid_loader = get_signal_train_validate_loader(batch_size=args.batch_size, shuffle=True)

    else:
        print("data error")

    defense_name = 'DD'
    dd = DistillationDefense(model=model_framework, defense_name=defense_name, dataset=dataset, temperature=args.temp,
                              device=device)
    dd.defense(initial_flag=args.initial, train_loader=train_loader, validation_loader=valid_loader, raw_train=raw_train_loader,
               raw_valid=raw_valid_loader, test_loader=test_loader)


if __name__ == '__main__':

    main()
