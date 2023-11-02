import math

import numpy as np
import torch

from Attacks.AttackMethods.AttackUtils import tensor2variable
from Attacks.AttackMethods.attacks import Attack


class UMIFGSMAttack(Attack):

    def __init__(self, model=None, epsilon=None, eps_iter=None, num_steps=5, decay_factor=1.0):
        """

        :param model:
        :param epsilon:
        :param eps_iter:
        :param num_steps:
        """
        super(UMIFGSMAttack, self).__init__(model)
        self.model = model

        self.epsilon = epsilon
        self.epsilon_iter = eps_iter
        self.num_steps = num_steps
        self.decay_factor = decay_factor

    def perturbation(self, samples, ys, device):
        """

        :param samples:
        :param ys:
        :param device:
        :return:
        """

        copy_samples = np.copy(samples)
        momentum = 0
        self.model.train()
        for index in range(self.num_steps):
            var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
            var_ys = tensor2variable(torch.LongTensor(ys), device=device)

            # obtain the gradient
            # self.model.eval()
            preds = self.model(var_samples)
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(preds, torch.max(var_ys, 1)[1])
            loss.backward()
            gradient = var_samples.grad.data.cpu().numpy()

            # update the momentum in the gradient direction
            momentum = self.decay_factor * momentum + gradient
            # update the (t+1) adversarial example
            copy_samples = copy_samples + self.epsilon_iter * np.sign(momentum)
            copy_samples = np.clip(copy_samples, samples - self.epsilon, samples + self.epsilon)
            copy_samples = np.clip(copy_samples, 0.0, 1.0)

        return copy_samples

    def batch_perturbation(self, xs, ys, batch_size, device):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param device:
        :return:
        """
        assert len(xs) == len(ys), "The lengths of samples and its ys should be equal"
        from Attacks.AttackMethods.AttackUtils import predict

        adv_sample = []
        adv_sample_all = []
        number_batch = int(math.ceil(len(xs) / batch_size))
        for index in range(number_batch):
            start = index * batch_size
            end = min((index + 1) * batch_size, len(xs))
            print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')

            batch_adv_images = self.perturbation(xs[start:end], ys[start:end], device)

            adv_sample.extend(batch_adv_images)
            adv_labels = predict(model=self.model, samples=batch_adv_images, device=device)
            adv_labels = torch.max(adv_labels, 1)[1]
            adv_labels = adv_labels.cpu().numpy()
            adv_sample_all.extend(adv_labels)
        return np.array(adv_sample), np.array(adv_sample_all)
