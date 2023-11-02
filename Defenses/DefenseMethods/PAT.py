import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
from Attacks.AttackMethods.AttackUtils import tensor2variable
from Defenses.DefenseMethods.defenses import Defense
from Utils.TrainTest import validation_evaluation
from models.network import define_model
from args import args
import shutil
import logging

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
if not os.path.exists(
        '../DefenseEnhancedModels/PAT/{}_{}only_eps{}/'.format(args.dataset, args.model,
                                                              args.eps)):
    os.makedirs(
        '../DefenseEnhancedModels/PAT/{}_{}only_eps{}/'.format(args.dataset, args.model,
                                                              args.eps))

defense_enhanced_saver = '../DefenseEnhancedModels/PAT/{}_{}only_eps{}/'.format(args.dataset, args.model,
                                                                               args.eps)

fh = logging.FileHandler(os.path.join(defense_enhanced_saver, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

num_epochs = 200
# batch_size=128
batch_size = args.batch_size  # vgg16
learning_rate = args.lr
momentum = .9
decay = 1e-6


class PATDefense(Defense):

    def __init__(self, model=None, defense_name=None, dataset=None, training_parameters=None, device=None, **kwargs):
        """

        :param model:
        :param defense_name:
        :param dataset:
        :param training_parameters:
        :param device:
        :param kwargs:
        """
        super(PATDefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device

        self.Dataset = dataset.upper()
        # assert self.Dataset in ['512', '3040'], "The data set must be 512 or 3040"

        # make sure to parse the parameters for the defense
        assert self._parsing_parameters(**kwargs)

        # get the training_parameters, the same as the settings of RawModels
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    def _parsing_parameters(self, **kwargs):

        assert kwargs is not None, "the parameters should be specified"

        print("\nparsing the user configuration for the {} defense".format(self.defense_name))
        for key in kwargs:
            print('\t{} = {}'.format(key, kwargs.get(key)))

        self.attack_step_num = kwargs.get('attack_step_num')
        self.step_size = kwargs.get('step_size')
        self.epsilon = kwargs.get('epsilon')

        return True

    def pgd_generation(self, var_natural_images=None, var_natural_labels=None):
        """

        :param var_natural_images:
        :param var_natural_labels:
        :return:
        """

        natural_images = var_natural_images.cpu().numpy()
        copy_images = natural_images.copy()
        copy_images = copy_images + np.random.uniform(-self.epsilon, self.epsilon, copy_images.shape).astype('float32')

        for i in range(self.attack_step_num):
            # self.model.eval()
            self.model.train()
            var_copy_images = torch.from_numpy(copy_images).to(self.device)
            var_copy_images.requires_grad = True

            preds = self.model(var_copy_images)
            loss = F.cross_entropy(preds, var_natural_labels)
            gradient = torch.autograd.grad(loss, var_copy_images)[0]
            gradient_sign = torch.sign(gradient).cpu().numpy()
            copy_images = copy_images + self.step_size * gradient_sign
            copy_images = np.clip(copy_images, natural_images - self.epsilon, natural_images + self.epsilon)
            copy_images = np.clip(copy_images, -1.0, 1.0)

        return torch.from_numpy(copy_images).to(self.device)

    def train_one_epoch_with_pgd_and_nat(self, train_loader, epoch):
        """
        :param train_loader:
        :param epoch:
        :return:
        """
        for index, (images, labels) in enumerate(train_loader):
            nat_images = images.to(self.device)
            nat_labels = labels.to(self.device)

            # prepare for adversarial examples using the pgd attack
            self.model.eval()
            adv_images = self.pgd_generation(var_natural_images=nat_images, var_natural_labels=nat_labels)

            # set the model in the training mode
            self.model.train()

            # logits_nat = self.model(nat_images)
            # loss_nat = F.cross_entropy(logits_nat, nat_labels)
            logits_adv = self.model(adv_images)
            loss_adv = F.cross_entropy(logits_adv, nat_labels)

            loss = loss_adv

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('\rTrain Epoch {:>3}: [{:>5}/{:>5}]  \t total_loss={:.4f} ===> '
                  .format(epoch, (index + 1) * len(images), len(train_loader) * len(images), loss),
                  end=' ')  # loss_adv={:.4f},loss_nat,loss_adv,loss_nat={:.4f},

    def defense(self, train_loader=None, validation_loader=None):

        best_val_acc = 0
        total = 0.0
        correct_cls = 0.0
        correct_adv = 0.0
        cls_losses = AverageMeter()
        adv_losses = AverageMeter()
        cls_loss = []
        adv_loss = []
        for epoch in range(self.num_epochs):

            # training the model with natural examples and corresponding adversarial examples
            adjust_lr(self.optimizer, epoch)
            # logging.info("epoch: ",epoch)
            self.train_one_epoch_with_pgd_and_nat(train_loader=train_loader, epoch=epoch)
            # val_acc = validation_evaluation(model=self.model, validation_loader=validation_loader, device=self.device)
            # adv_images = self.pgd_generation(var_natural_images=, var_natural_labels=nat_labels)

            for i, (inputs, labels) in enumerate(validation_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                adv_inputs = self.pgd_generation(var_natural_images=inputs, var_natural_labels=labels)
                outputs = self.model(inputs)
                outputs_adv = self.model(adv_inputs)
                _, predicted_cls = torch.max(outputs.data, 1)
                _, predicted_adv = torch.max(outputs_adv.data, 1)
                total = total + labels.size(0)
                correct_cls = correct_cls + (predicted_cls == labels).sum().item()
                correct_adv = correct_adv + (predicted_adv == labels).sum().item()
                logits_nat = self.model(inputs)
                loss_nat = F.cross_entropy(logits_nat, labels)
                logits_adv = self.model(adv_inputs)
                loss_adv = F.cross_entropy(logits_adv, labels)

                cls_losses.update(loss_nat.item(), inputs.size(0))
                adv_losses.update(loss_adv.item(), inputs.size(0))

            ratio_cls = correct_cls / total
            ratio_adv = correct_adv / total
            logs = ('acc:{0}    '
                    'adv_acc:{1}    '
                    'cls_loss:{2}   '
                    'adv_loss:{3}   '.format(ratio_cls, ratio_adv, cls_losses.avg, adv_losses.avg))

            cls_loss.append(cls_losses.avg)
            adv_loss.append(adv_losses.avg)

            np.save(defense_enhanced_saver + 'cls.npy', cls_loss)
            np.save(defense_enhanced_saver + 'adv.npy', adv_loss)

            logging.info('Epoch: {} '.format(epoch))
            # logging.info('train_acc:{acc}'.format(acc=ratio_cls), 'train_adv_acc:{acc}'.format(acc=ratio_adv))
            logging.info(logs)

            is_best = False
            if best_val_acc < ratio_cls:
                best_val_acc = ratio_cls
                is_best = True
            else:
                print('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch,
                                                                                                           best_val_acc))
                logging.info('Train Epoch{:>3}: validation dataset accuracy did not improve from {:.4f}\n'.format(epoch,
                                                                                                                  best_val_acc))
            if is_best:
                logging.info('Saving models......')
            save_checkpoint({
                'epoch': epoch,
                'net': self.model.state_dict(),
                'prec@1': best_val_acc,
                # 'prec@5': test_top5,
            }, is_best, defense_enhanced_saver, epoch)
            logging.info('best_train_acc:{acc}'.format(acc=best_val_acc))


def save_checkpoint(state, is_best, save_root, epoch):
    save_path = os.path.join(save_root, 'checkpoint.pth_{}.tar'.format(epoch))
    torch.save(state, save_path)
    if is_best:
        best_save_path = os.path.join(save_root, 'model_best.pth_epoch{}.tar'.format(epoch))
        shutil.copyfile(save_path, best_save_path)


def adjust_lr(optimizer, epoch):
    scale = 0.1
    lr_list = [args.lr] * 100
    lr_list += [args.lr * scale] * 50
    lr_list += [args.lr * scale * scale] * 50

    lr = lr_list[epoch]
    # logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print('The **learning rate** of the {} epoch is {}'.format(epoch, param_group['lr']))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
