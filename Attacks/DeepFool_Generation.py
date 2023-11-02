import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))

from Attacks.AttackMethods.AttackUtils import predict
from Attacks.AttackMethods.DEEPFOOL import DeepFoolAttack
from Attacks.Generation import Generation
from args import args
from Generation import write_result_to_csv

class DeepFoolGeneration(Generation):
    def __init__(self, dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir, device, overshoot, max_iters):
        super(DeepFoolGeneration, self).__init__(dataset, attack_name, targeted, raw_model_location, clean_data_location, adv_examples_dir,
                                                 device)

        self.overshoot = overshoot
        self.max_iters = max_iters

    def generate(self):
        attacker = DeepFoolAttack(model=self.raw_model, overshoot=self.overshoot, max_iters=self.max_iters)
        adv_samples= attacker.perturbation(xs=self.nature_samples, device=self.device)
        # prediction for the adversarial examples  , adv_labels
        adv_labels = predict(model=self.raw_model, samples=adv_samples, device=self.device)
        adv_labels = torch.max(adv_labels, 1)[1]
        adv_labels = adv_labels.cpu().numpy()

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
                                                                          int(len(adv_samples) - mis), len(adv_samples),
                                                                          int(len(adv_samples) - mis) / len(
                                                                              adv_labels) * 100))
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

    name = 'DeepFool'
    targeted = False

    df = DeepFoolGeneration(dataset=args.dataset, attack_name=name, targeted=targeted, raw_model_location=args.modelDir,
                            clean_data_location=args.cleanDir, adv_examples_dir=args.adv_saver, device=device, max_iters=args.max_iters,
                            overshoot=args.overshoot)
    mr, len = df.generate()
    write_result_to_csv(
        model=args.model,
        dataset=args.dataset,
        snr=args.db,
        note=args.note,
        attack=name,
        number=len,
        robustacc = 100-mr
    )

if __name__ == '__main__':
    main()
