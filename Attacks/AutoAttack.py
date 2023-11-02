import os
import argparse
import pathlib
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

import sys
sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from models.network import define_model
from utils import load_pretrained_model
from sklearn.preprocessing import OneHotEncoder
from args import args

# sys.path.insert(0, '..')


# write results to csv
def write_result_to_csv(**kwargs):
    results = pathlib.Path("../AdversarialExampleDatasets") / "results.csv"

    if not results.exists():
        results.write_text(
            "MODEL, "
            "DATA, "
            "SNR, "
            "NOTE, "
            "ATTACK, "
            "#ADV-DATA, "
            "Robust-ACC\n "
        )

    with open(results, "a+") as f:  # a+附加读写方式打开
        f.write(
            ("{model}, "
             "{dataset}, "
             "{snr}, "
             "{note}, "
             "{attack}, "
             "{number}, "
             "{robustacc}\n"
             ).format(**kwargs)
        )


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
    device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
    print("CUDA:", args.gpu_index)

    # load model
    # raw_model_location = "/home/zjut/public/signal/wzw/KD/DefenseEnhancedModels/PAT/128_CNN1Donly_eps0.06/model_best.pth_epoch199.tar"
    # raw_model_location = "/home/zjut/public/signal/wzw/KD/results/Student_model/CNN1D_AMD_200.tar"
    raw_model_location = args.location
    raw_model = define_model(name=args.model)
    # print(raw_model.state_dict().keys())
    # raw_model.load_state_dict(torch.load(raw_model_location,map_location=f"cuda:{args.gpu_index }"))#["net"]
    checkpoint = torch.load(raw_model_location, map_location='cuda:{}'.format(args.gpu_index))
    load_pretrained_model(raw_model, checkpoint['net'])
    # load_pretrained_model(raw_model, checkpoint)

    raw_model.eval()

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)

    # item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    # test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

    # test_loader = get_testloader(batch_size=128, shuffle=False, num_worker=1)

    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load attack
    from autoattack import AutoAttack

    adversary = AutoAttack(raw_model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
                           version=args.version, device=device)

    x_test = np.load('/home/zjut/public/signal/wzw/KD/CleanDatasets/{}_{}/128/128_inputs.npy'.format(args.model, args.note))
    y_test = np.load('/home/zjut/public/signal/wzw/KD/CleanDatasets/{}_{}/128/128_labels.npy'.format(args.model, args.note))
    y_test = np.argmax(y_test, axis=1)

    x_test = torch.tensor(x_test).to(device)
    y_test = torch.tensor(y_test).to(device)

    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2

    print("AA eps: ", args.epsilon)
    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete, robust_accuracy = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                                                             bs=args.batch_size, state_path=args.state_path)

            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.3f}.pth'.format(
                args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                                                                        y_test[:args.n_ex], bs=args.batch_size)

            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))

    write_result_to_csv(
        model=args.model,
        dataset=args.dataset,
        snr=args.db,
        note=args.note,
        attack="AA",
        number=len,
        robustacc=robust_accuracy*100
    )

