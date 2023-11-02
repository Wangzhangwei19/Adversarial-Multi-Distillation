import copy
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomAffine, RandomHorizontalFlip, ToPILImage, ToTensor

from Defenses.DefenseMethods.defenses import Defense
from Utils.TrainTest import testing_evaluation, validation_evaluation
from args import args
import shutil
import logging
log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
if not os.path.exists('../DefenseEnhancedModels/DD/{}_{}_temp{}/'.format(args.dataset, args.model, args.temp)): #temp20
    os.makedirs('../DefenseEnhancedModels/DD/{}_{}_temp{}/'.format(args.dataset, args.model, args.temp))
defense_enhanced_saver = '../DefenseEnhancedModels/DD/{}_{}_temp{}/'.format(args.dataset, args.model, args.temp)
fh = logging.FileHandler(os.path.join(defense_enhanced_saver, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# For help when requiring self-defined dataset provision with batches during training
class SoftLabelDataset(Dataset):
    def __init__(self, images, labels, dataset, transform):
        super(SoftLabelDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.transform = transform
        # self.color_mode = 'RGB' if dataset == 'CIFAR10' else 'L'

    def __getitem__(self, index):
        single_image, single_label = self.images[index], self.labels[index]
        # if self.transform:
            # img = ToPILImage(mode=self.color_mode)(single_image)
            # single_image = self.transform(img)
        return single_image, single_label

    def __len__(self):
        return len(self.images)

num_epochs= 200
batch_size=512
learning_rate=args.lr
# learning_rate=0.001
momentum=.9
decay=1e-6

class DistillationDefense(Defense):

    def __init__(self, model=None, defense_name=None, dataset=None, temperature=1, device=None):
        """

        :param model:
        :param defense_name:
        :param dataset:
        :param temperature:
        :param training_parameters:
        :param device:
        """
        super(DistillationDefense, self).__init__(model=model, defense_name=defense_name)
        self.model = model
        self.defense_name = defense_name
        self.device = device

        self.Dataset = dataset.upper()
        # assert self.Dataset in ['MNIST', 'CIFAR10'], "The data set must be MNIST or CIFAR10"

        # prepare the models for the defenses
        self.initial_model = copy.deepcopy(model)
        self.best_initial_model = copy.deepcopy(model)
        self.distilled_model = copy.deepcopy(model)

        # parameters for the defense
        self.temperature = temperature * 1.0

        # get the training_parameters, the same as the settings of RawModels
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # prepare the optimizers and transforms
        if dataset == '128' or '512' or '1024' or '3040':
            self.initial_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.initial_model.parameters()), lr=args.lr)
            self.distilled_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.distilled_model.parameters()), lr=args.lr)
            self.transform = None
        else:
            print()
            # self.initial_optimizer = optim.Adam(self.initial_model.parameters(), lr=training_parameters['lr'])
            # self.distilled_optimizer = optim.Adam(self.distilled_model.parameters(), lr=training_parameters['lr'])
            #数据增强
            # self.transform = Compose([RandomAffine(degrees=0, translate=(0.1, 0.1)), RandomHorizontalFlip(), ToTensor()])

    def train_initial_model_with_temperature(self, train_loader=None, validation_loader=None):
        """

        :param train_loader:
        :param validation_loader:
        :return:
        """
        print("\nTraining the initial model ......\n")
        best_val_acc = None
        for epoch in range(self.num_epochs):
            self.initial_model.train()  # set the model in the train mode before every epoch
            for index, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward the NN with the temperature
                logits = self.initial_model(images)
                logits_with_temp = logits / self.temperature
                loss = F.cross_entropy(logits_with_temp, labels)

                # backward
                self.initial_optimizer.zero_grad()
                loss.backward()
                self.initial_optimizer.step()

                print('\rTrain Epoch {:>3}: [batch:{:>4}/{:>4}({:>3.0f}%)]  \tLoss: {:.4f} ===> '. \
                      format(epoch, index, len(train_loader), index / len(train_loader) * 100.0, loss.item()), end=' ')

            # validation
            val_acc = validation_evaluation(model=self.initial_model, validation_loader=validation_loader, device=self.device)

            adjust_lr(self.initial_optimizer, epoch)


            # save the initial model
            assert os.path.exists('../DefenseEnhancedModels/{}'.format(self.defense_name))
            initial_model_saver = '../DefenseEnhancedModels/{}/{}_DD_initial.pt'.format(self.defense_name, self.Dataset)
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                if best_val_acc is not None:
                    os.remove(initial_model_saver)
                best_val_acc = val_acc
                self.initial_model.save(name=initial_model_saver)
            else:
                print('Train Epoch {:>3}: validation dataset accuracy of *Initial Model* did not improve from {:.4f}\n'. \
                      format(epoch, best_val_acc))

    def train_distilled_model_with_temp(self, distilled_train_loader=None, validation_loader=None):
        """

        :param distilled_train_loader:
        :param validation_loader:
        :return:
        """
        print("\nTraining distilled model ......\n")
        best_val_acc = None
        for epoch in range(self.num_epochs):
            adjust_lr(self.distilled_optimizer, epoch)
            self.distilled_model.train()  # set the model in the train mode before every epoch
            for index, (images, labels) in enumerate(distilled_train_loader):
                images.requires_grad = True

                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.distilled_model(images)

                eps = 1e-20  # small value to avoid evaluation of log(0)
                # eps = 0  # small value to avoid evaluation of log(0)
                logits_with_temp = (logits + eps) / self.temperature
                # cross entropy for soft labels
                log_likelihood = - F.log_softmax(logits_with_temp, dim=1)
                loss = torch.sum(torch.mul(log_likelihood, labels), dim=1)
                loss = torch.mean(loss)

                self.distilled_optimizer.zero_grad()
                loss.backward()
                self.distilled_optimizer.step()

                print('\rTrain Epoch {:>3}: [batch:{:>4}/{:>4}({:>3.0f}%)]  \tLoss: {:.4f} ===> '. \
                      format(epoch, index, len(distilled_train_loader), index / len(distilled_train_loader) * 100.0, loss.item()), end=' ')

            val_acc = validation_evaluation(model=self.distilled_model, validation_loader=validation_loader, device=self.device)

            logging.info('Epoch: {} '.format(epoch))
            logging.info('train_acc:{acc}'.format(acc=val_acc))
            is_best = False
            if not best_val_acc or round(val_acc, 4) >= round(best_val_acc, 4):
                # if best_val_acc is not None:
                    # os.remove(distilled_model_saver)
                best_val_acc = val_acc
                # self.distilled_model.save(name=distilled_model_saver)
                is_best = True
            else:
                print('Train Epoch {:>3}: validation dataset accuracy of *Distilled Model* did not improve from {:.4f}\n'. \
                      format(epoch, best_val_acc))
            if is_best:
                logging.info('Saving models......')
            save_checkpoint({
                'epoch': epoch,
                'net': self.distilled_model.state_dict(),
                'prec@1': best_val_acc,
                # 'prec@5': test_top5,
            }, is_best, defense_enhanced_saver, epoch)
            logging.info('best_train_acc:{acc}'.format(acc=best_val_acc))


    def defense(self, initial_flag=None, train_loader=None, validation_loader=None, raw_train=None, raw_valid=None, test_loader=None):
        """

        :param initial_flag: whether there is the initial model or not
        :param train_loader: train dataset loader for training the initial model
        :param validation_loader: train validation loader for the initial model
        :param raw_train: raw train dataset loader used to construct the dataset with soft label for training the distilled model
        :param raw_valid: ... for validating the distilled model
        :param test_loader: test dataset loader for testing models
        :return:
        """
        if initial_flag is False:
            # train the initial model
            self.train_initial_model_with_temperature(train_loader=train_loader, validation_loader=validation_loader)

        # load the pre-trained initial model
        # initial model location
        # model_location = '/home/zjut/public/signal/wzw/project_attack/Radio_Adversarial_Attacks/result/model/{}_{}_best_lr=0.001.pth'\
        #     .format(args.dataset, args.model)
        model_location = '/home/zjut/public/signal/wzw/KD/results/all-db-baseline/model/{}_{}_best_lr=0.001.pth'\
            .format(args.dataset, args.model)
        assert os.path.exists(model_location), 'No initial model, please train the initial model first'
        # self.best_initial_model.load(path=model_location, device=self.device)
        self.best_initial_model.load_state_dict(torch.load(model_location, map_location=f"cuda:{args.gpu_index}"))  # ["net"]

        # prepare the training data set with soft labels for the distilled model training
        self.best_initial_model.eval()
        with torch.no_grad():
            ori_images = []
            soft_labels = []
            for images, _ in raw_train:
                images = images.to(self.device)
                initial_logits = self.best_initial_model(images)
                initial_logits_temp = initial_logits / self.temperature
                initial_preds = F.softmax(initial_logits_temp, dim=1).cpu().detach().numpy()

                ori_images.extend(images.cpu().numpy())
                soft_labels.extend(initial_preds)

            ori_images = np.array(ori_images)
            soft_labels = np.array(soft_labels)
        soft_dataset = SoftLabelDataset(images=torch.from_numpy(ori_images), labels=torch.from_numpy(soft_labels), dataset=self.Dataset,
                                        transform=self.transform)
        soft_dataset_loader = torch.utils.data.DataLoader(soft_dataset, batch_size=self.batch_size, shuffle=True)

        # train and save the distilled model -> DD_enhanced model
        self.train_distilled_model_with_temp(distilled_train_loader=soft_dataset_loader, validation_loader=validation_loader)

def save_checkpoint(state, is_best, save_root, epoch):
    save_path = os.path.join(save_root, 'checkpoint.pth_{}.tar'.format(epoch))
    torch.save(state, save_path)
    if is_best:
        best_save_path = os.path.join(save_root, 'model_best.pth_epoch{}.tar'.format(epoch))
        shutil.copyfile(save_path, best_save_path)

def adjust_lr(optimizer, epoch):
    scale   = 0.1
    lr_list =  [args.lr] * 100
    lr_list += [args.lr*scale] * 50
    lr_list += [args.lr*scale*scale] * 50

    lr = lr_list[epoch]
    # logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print('The **learning rate** of the {} epoch is {}'.format(epoch, param_group['lr']))