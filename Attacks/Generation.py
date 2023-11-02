import os
import shutil
from abc import ABCMeta

import numpy as np
import torch
import pathlib
from args import args
from models.network import define_model
from utils import load_pretrained_model, save_checkpoint


class Generation(object):
    __metaclass__ = ABCMeta

    def __init__(self, dataset='128', attack_name='FGSM', targeted=False, raw_model_location='../RawModels/',
                 clean_data_location='../CleanDatasets/', adv_examples_dir='../AdversarialExampleDatasets/',
                 device=torch.device('cpu')):
        # check and set the support data set
        self.dataset = dataset.upper()
        if self.dataset not in {'128', '512', '1024', '3040'}:
            raise ValueError("The data set must be 128 or 512 or 1024 or 3040 ")

        # check and set the supported attackS
        self.model = args.model
        self.attack_name = attack_name.upper()
        supported = {'FGSM', 'RFGSM', 'BIM', 'PGD', 'UMIFGSM', 'UAP', 'DEEPFOOL', 'OM', 'LLC', "RLLC", 'ILLC',
                     'TMIFGSM', 'JSMA', 'BLB', 'CW2',
                     'EAD'}
        if self.attack_name not in supported:
            raise ValueError(
                self.attack_name + 'is unknown!\nCurrently, our implementation support the attacks: ' + ', '.join(
                    supported))

        # load the raw model
        # raw_model_location = "/home/zjut/public/signal/wzw/KD/DefenseEnhancedModels/PAT/1024_CNN1Donly_eps0.06/checkpoint.pth_199.tar"
        raw_model_location = args.location
        if dataset == '128' or '512' or '1024' or '3040':
            self.raw_model = define_model(name=args.model)
            # self.raw_model.load_state_dict(torch.load(raw_model_location,map_location=f"cuda:{args.gpu_index }"))

            checkpoint = torch.load(raw_model_location, map_location='cuda:{}'.format(args.gpu_index))
            # load_pretrained_model(self.raw_model, checkpoint)  # ['net']
            load_pretrained_model(self.raw_model, checkpoint['net']) #

        else:
            print("Data error")

        # get the clean data sets / true_labels / targets (if the attack is one of the targeted attacks)
        print(
            'Loading the prepared clean samples (nature inputs and corresponding labels) that will be attacked ...... ')
        self.nature_samples = np.load(
            '{}{}/{}/{}_inputs.npy'.format(clean_data_location, self.model, self.dataset, self.dataset))
        self.labels_samples = np.load(
            '{}{}/{}/{}_labels.npy'.format(clean_data_location, self.model, self.dataset, self.dataset))

        # self.nature_samples = np.load("/home/zjut/public/signal/wzw/KD/CleanDatasets/CNN1D_AMD/128/128_inputs.npy")
        # self.labels_samples = np.load("/home/zjut/public/signal/wzw/KD/CleanDatasets/CNN1D_AMD/128/128_labels.npy")

        if targeted:
            print('For Targeted Attacks, loading the randomly selected targeted labels that will be attacked ......')
            if self.attack_name.upper() in ['LLC', 'RLLC', 'ILLC']:
                print('#### Especially, for LLC, RLLC, ILLC, loading the least likely class that will be attacked')
                self.targets_samples = np.load(
                    '{}{}/{}/{}_llc.npy'.format(clean_data_location, self.model, self.dataset, self.dataset))
            else:
                self.targets_samples = np.load(
                    '{}{}/{}/{}_targets.npy'.format(clean_data_location, self.model, self.dataset, self.dataset))

        # prepare the directory for the attacker to save their generated adversarial examples
        self.adv_examples_dir = adv_examples_dir + self.model + '/' + self.dataset + '/' + self.attack_name + '/'
        if self.model not in os.listdir(adv_examples_dir):
            os.mkdir(adv_examples_dir + self.model + '/')

        if self.dataset not in os.listdir(adv_examples_dir + self.model + '/'):
            os.mkdir(adv_examples_dir + self.model + '/' + self.dataset + '/')

        if self.attack_name not in os.listdir(adv_examples_dir + self.model + '/' + self.dataset + '/'):
            os.mkdir(adv_examples_dir + self.model + '/' + self.dataset + '/' + self.attack_name + '/')

        else:
            shutil.rmtree('{}'.format(self.adv_examples_dir))
            os.mkdir(self.adv_examples_dir)

        # set up device
        self.device = device

        # write_result_to_csv(
        #     model = args.model,
        #     dataset = args.dataset,
        #     number = args.number,
        #     mis = BIMGeneration
        # )

    def generate(self):
        print("abstract method of Generation is not implemented")
        raise NotImplementedError


# def calculateSNR()


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
