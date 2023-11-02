import math

import numpy as np
import torch

from Attacks.AttackMethods.AttackUtils import tensor2variable
from Attacks.AttackMethods.attacks import Attack


class PGDAttack(Attack):

    def __init__(self, model=None, epsilon=None, eps_iter=None, num_steps=5):
        """

        :param model:
        :param epsilon:
        :param eps_iter:
        :param num_steps:
        """
        super(PGDAttack, self).__init__(model)
        self.model = model
        self.epsilon = epsilon
        self.epsilon_iter = eps_iter
        self.num_steps = num_steps

    def perturbation(self, samples, ys, device):
        """

        :param samples:
        :param ys:
        :param device:
        :return:
        """

        copy_samples = np.copy(samples)
        self.model.to(device)
        # randomly chosen starting points inside the L_\inf ball around the
        copy_samples = copy_samples + np.random.uniform(-self.epsilon, self.epsilon, copy_samples.shape).astype('float32')
        self.model.train()
        for index in range(self.num_steps):
            var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
            var_ys = tensor2variable(torch.LongTensor(ys), device=device)

            # self.model.eval()
            preds = self.model(var_samples)
            loss_fun = torch.nn.CrossEntropyLoss()
            loss = loss_fun(preds, torch.max(var_ys, 1)[1])
            loss.backward()

            gradient_sign = var_samples.grad.data.cpu().sign().numpy()
            copy_samples = copy_samples + self.epsilon_iter * gradient_sign

            copy_samples = np.clip(copy_samples, samples - self.epsilon, samples + self.epsilon)
            copy_samples = np.clip(copy_samples, -1.0, 1.0)

            # copy_samples = torch.clip(copy_samples, samples - self.epsilon, samples + self.epsilon)
            # copy_samples = torch.clip(copy_samples,-1.0, 1.0)

        return copy_samples

    def batch_perturbation(self, xs, ys, batch_size, device):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param device:
        :return:
        """
        from Attacks.AttackMethods.AttackUtils import predict
        assert len(xs) == len(ys), "The lengths of samples and its ys should be equal"

        adv_sample = []
        adv_labels_all = []
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
            adv_labels_all.extend(adv_labels)
        # adv_sample = np.array(adv_sample)

        return np.array(adv_sample), np.array(adv_labels_all)
        # return adv_sample
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References:   A. Madry, et al., "Towards deep learning models resistant to adversarial attacks," in ICLR, 2018.
# Reference Implementation from Authors (TensorFlow):
#       [2] https://github.com/MadryLab/mnist_challenge
#       [3] https://github.com/MadryLab/cifar10_challenge
# **************************************
# @Time    : 2018/9/9 1:27
# @Author  : Xiang Ling
# @Lab     : nesa.zju.edu.cn
# @File    : PGD.py
# **************************************

# import math
#
# import numpy as np
# import torch
#
# from Attacks.AttackMethods.AttackUtils import tensor2variable
# from Attacks.AttackMethods.attacks import Attack
#
#
# class PGDAttack(Attack):
#
#     def __init__(self, model=None, epsilon=None, eps_iter=None, num_steps=5):
#         """
#
#         :param model:
#         :param epsilon:
#         :param eps_iter:
#         :param num_steps:
#         """
#         super(PGDAttack, self).__init__(model)
#         self.model = model
#         self.epsilon = epsilon
#         self.epsilon_iter = eps_iter
#         self.num_steps = num_steps
#
#     def perturbation(self, samples, ys, device):
#         """
#
#         :param samples:
#         :param ys:
#         :param device:
#         :return:
#         """
#
#         copy_samples = np.copy(samples)
#         self.model.to(device)
#         # randomly chosen starting points inside the L_\inf ball around the
#         copy_samples = copy_samples + np.random.uniform(-self.epsilon, self.epsilon, copy_samples.shape).astype('float32')
#         self.model.train()
#         for index in range(self.num_steps):
#             var_samples = tensor2variable(torch.from_numpy(copy_samples), device=device, requires_grad=True)
#             var_ys = tensor2variable(torch.LongTensor(ys), device=device)
#
#             # self.model.eval()
#             preds = self.model(var_samples)
#             loss_fun = torch.nn.CrossEntropyLoss()
#             loss = loss_fun(preds, torch.max(var_ys, 1)[1])
#             loss.backward()
#
#             gradient_sign = var_samples.grad.data.cpu().sign().numpy()
#             copy_samples = copy_samples + self.epsilon_iter * gradient_sign
#
#             copy_samples = np.clip(copy_samples, samples - self.epsilon, samples + self.epsilon)
#             copy_samples = np.clip(copy_samples,-1.0, 1.0)
#
#         return copy_samples
#
#     def batch_perturbation(self, xs, ys, batch_size, device):
#         """
#
#         :param xs:
#         :param ys:
#         :param batch_size:
#         :param device:
#         :return:
#         """
#         from Attacks.AttackMethods.AttackUtils import predict
#         assert len(xs) == len(ys), "The lengths of samples and its ys should be equal"
#
#         adv_sample = []
#         adv_labels_all = []
#         number_batch = int(math.ceil(len(xs) / batch_size))
#
#         for index in range(number_batch):
#             start = index * batch_size
#             end = min((index + 1) * batch_size, len(xs))
#             print('\r===> in batch {:>2}, {:>4} ({:>4} in total) nature examples are perturbed ... '.format(index, end - start, end), end=' ')
#
#             batch_adv_images = self.perturbation(xs[start:end], ys[start:end], device)
#
#             adv_sample.extend(batch_adv_images)
#
#             adv_labels = predict(model=self.model, samples=batch_adv_images, device=device)
#             adv_labels = torch.max(adv_labels, 1)[1]
#             adv_labels = adv_labels.cpu().numpy()
#             adv_labels_all.extend(adv_labels)
#         # adv_sample = np.array(adv_sample)
#
#         return np.array(adv_sample), np.array(adv_labels_all)
#         # return adv_sample
