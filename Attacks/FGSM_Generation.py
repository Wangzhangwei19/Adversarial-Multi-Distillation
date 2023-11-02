import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from Attacks.AttackMethods.AttackUtils import predict
from Attacks.AttackMethods.FGSM import FGSMAttack
from Attacks.Generation import Generation
from args import args
from Generation import write_result_to_csv


class FGSMGeneration(Generation):

    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir, device, eps,
                 attack_batch_size):
        super(FGSMGeneration, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir, device)
        self.attack_batch_size = attack_batch_size

        self.epsilon = eps

    def generate(self):
        attacker = FGSMAttack(model=self.raw_model, epsilon=self.epsilon)

        # generating
        adv_samples, adv_labels = attacker.batch_perturbation(xs=self.nature_samples, ys=self.labels_samples, batch_size=self.attack_batch_size,
                                                  device=self.device)

        np.save('{}{}_AdvExamples.npy'.format(self.adv_examples_dir, self.attack_name), adv_samples)
        np.save('{}{}_AdvLabels.npy'.format(self.adv_examples_dir, self.attack_name), adv_labels)
        np.save('{}{}_TrueLabels.npy'.format(self.adv_examples_dir, self.attack_name), self.labels_samples)

        mis = 0
        for i in range(len(adv_samples)):
            if self.labels_samples[i].argmax(axis=0) != adv_labels[i]:
                mis = mis + 1
        print('\nFor **{}** on **{}**: misclassification ratio is {}/{}={:.1f}%\n'.format(self.attack_name, self.dataset, mis, len(adv_samples),
                                                                                          mis / len(adv_labels) * 100))
        print('\nFor **{}** on **{}**: adv ACC is {}/{}={:.1f}%\n'.format(self.attack_name, self.dataset,
                                                                                        int(len(adv_samples)-mis), len(adv_samples),
                                                                                        int(len(adv_samples)-mis) / len(adv_labels) * 100))
        return mis / len(adv_labels) * 100, len(adv_labels)

def main():
    # Device configuration
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

    name = 'FGSM'
    targeted = False
    fgsm = FGSMGeneration(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                          clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device, eps=args.epsilon,
                          attack_batch_size=args.attack_batch_size)
    mr, len = fgsm.generate()

    write_result_to_csv(
        model=args.model,
        dataset=args.dataset,
        snr=args.db,
        note=args.note,
        attack=name,
        number=len,
        robustacc=100-mr
    )


if __name__ == '__main__':
    main()
