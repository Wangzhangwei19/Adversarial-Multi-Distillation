import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from Utils.dataset import get_signal_train_validate_loader, get_signal_test_loader

from Utils.dataset import get_signal_test_loader
from models.network import define_model
from Defenses.DefenseMethods.PAT import PATDefense
from Defenses.DefenseMethods.DD import DistillationDefense
from args import args


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
    device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
    print("CUDA:", args.gpu_index)
    # Set the random seed manually for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Get training parameters, set up model frameworks and then get the train_loader and test_loader
    dataset = args.dataset.upper()

    if dataset == '128' or '512' or '1024' or '3040':
        model_framework = define_model(name=args.model).to(device)
        train_loader, valid_loader = get_signal_train_validate_loader(batch_size=args.batch_size,shuffle=True)
        # testing dataset loader
        test_loader = get_signal_test_loader(batch_size=args.batch_size, shuffle=False)
        # raw train_loader (no augmentation) for constructing the SoftLabelDataset and then used to train the distilled model
        raw_train_loader, raw_valid_loader = get_signal_train_validate_loader(batch_size=args.batch_size, shuffle=True)
    else:
        print("data error")

    if args.defense_name == 'PAT':
        defense_name = 'PAT'
        pat_params = {
            'attack_step_num': args.step_num,
            'step_size': args.step_size,
            'epsilon': args.eps
        }
        pat = PATDefense(model=model_framework, defense_name=defense_name, dataset=dataset, device=device, **pat_params)
        pat.defense(train_loader=train_loader, validation_loader=valid_loader)
    elif args.defense_name == 'DD':
        defense_name = 'DD'
        dd = DistillationDefense(model=model_framework, defense_name=defense_name, dataset=dataset,
                                 temperature=args.temp,
                                 device=device)
        dd.defense(initial_flag=args.initial, train_loader=train_loader, validation_loader=valid_loader,
                   raw_train=raw_train_loader,
                   raw_valid=raw_valid_loader, test_loader=test_loader)
    elif args.defense_name == 'VMD':
        VMDdefense()

if __name__ == '__main__':
    main()
