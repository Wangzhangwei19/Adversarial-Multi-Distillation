import argparse
import sys
import yaml
from pathlib import Path

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description='Candidate Selection for Clean Data set')
    parser.add_argument('--db', type=int, required=False, help='single db test')
#kd
    parser.add_argument('--t1_model', type=str, required=False, help='path name of teacher model',
                        default='/home/zjut/public/signal/wzw/KD/results/EXP_model/all-db-baseline/model/128_Vgg16_best_lr=0.001.pth')
    parser.add_argument('--t2_model', type=str, required=False, help='path name of teacher model',
                        default='/home/zjut/public/signal/wzw/SignalAttack/DefenseEnhancedModels/PAT_ALLdB_model/128_Vgg16mmmoo1/model_best.pth.tar')
    parser.add_argument('--img_root', type=str, default='./datasets', help='path name of image dataset')
    parser.add_argument('--s_init', type=str, required=False, help='initial parameters of student model')
    parser.add_argument('--kd_mode', type=str, required=False)
    parser.add_argument('--lambda_kd1', type=float, default=1.0, help='trade-off parameter for kd loss')
    parser.add_argument('--lambda_kd2', type=float, default=1.0, help='trade-off parameter for kd loss')
    parser.add_argument('--lambda_kd3', type=float, default=1.0, help='trade-off parameter for kd loss')
    parser.add_argument('--lambda_kd0', type=float, default=1.0, help='trade-off parameter for kd loss')
    parser.add_argument('--s_name', type=str, default='Lenet', help='the model ')
    parser.add_argument('--t1_name', type=str, default='Vgg16', help='the model ')
    parser.add_argument('--t2_name', type=str, default='Vgg16', help='the model ')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')

    parser.add_argument('--step', type=int, default=5, help='step num of PAT')
    parser.add_argument('--T', type=float, default=4.0, help='temperature for ST')
    # parser.add_argument('--step_size', type=float, default=0.03, help='evertime attack epsilon')
    # parser.add_argument('--img_root', type=str, default='../data/cifar10', help='path name of image dataset')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--cuda', type=int, default=1)

    # others
    parser.add_argument('--note', type=str, default='try', help='note for this run')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
    # net and dataset choosen
    parser.add_argument('--data_name', type=str, help='name of dataset',
                        default='128')  # cifar10/cifar100 required=True,
    parser.add_argument('--net_name', type=str, help='name of basenet',
                        default='r8conv1')  # resnet20/resnet110 required=True,
    # 干净样本 攻击 模型位置读取
    parser.add_argument('--location', type=str, help='model attack test model load location')
#clean data selection
    parser.add_argument('--model', type=str, default='r8conv1', help='the model ')
    parser.add_argument('--name', type=str, default='r8conv1', help='the model ')
    parser.add_argument('--dataset', type=str, default='128', help='the dataset ')
    parser.add_argument('--seed', type=int, default=100, help='the default random seed for numpy and torch')
    parser.add_argument('--gpu_index', type=str, default='0', help="gpu index to use")
    parser.add_argument('--number', type=int, default=1000, help='the total number of candidate samples that will be randomly selected')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
#ATTACK METHOD
    parser.add_argument('--modelDir', type=str, default='../RawModels/', help='the directory for the raw model')
    parser.add_argument('--cleanDir', type=str, default='../CleanDatasets/', help='the directory for the clean dataset that will be attacked')
    parser.add_argument('--adv_saver', type=str, default='../AdversarialExampleDatasets/',
                        help='the directory used to save the generated adversarial examples')
    # FGSM
    parser.add_argument('--epsilon', type=float, default=0.06, help='the epsilon value of FGSM')

    # PGD
    parser.add_argument('--epsilon_iter', type=float, default=0.03, help='the one iterative eps of PGD')
    parser.add_argument('--num_steps', type=int, default=5, help='the number of perturbation steps')
    parser.add_argument('--attack_batch_size', type=int, default=100, help='the default batch size for adversarial example generation')

    #DEEPFOOL
    parser.add_argument('--max_iters', type=int, default=15, help="the max iterations")#50
    parser.add_argument('--overshoot', type=float, default=0.02, help='the overshoot')

    #UAP
    parser.add_argument('--fool_rate', type=float, default=1.0, help="the fooling rate")
    parser.add_argument('--max_iter_universal', type=int, default=20, help="the maximum iterations for UAP")
    parser.add_argument('--max_iter_deepfool', type=int, default=10, help='the maximum iterations for DeepFool')

    #UMIFGSM
    parser.add_argument('--decay_factor', type=float, default=1.0, help='decay factor')

    #CW2
    parser.add_argument('--confidence', type=float, default=0, help='the confidence of adversarial examples')
    parser.add_argument('--initial_const', type=float, default=0.001,
                        help="the initial value of const c in the binary search.")
    parser.add_argument('--learning_rate', type=float, default=0.02, help="the learning rate of gradient descent.")
    parser.add_argument('--iteration', type=int, default=10000, help='maximum iteration')
    parser.add_argument('--lower_bound', type=float, default=0.0,
                        help='the minimum pixel value for examples (default=0.0).')
    parser.add_argument('--upper_bound', type=float, default=1.0,
                        help='the maximum pixel value for examples (default=1.0).')
    parser.add_argument('--search_steps', type=int, default=10,
                        help="the binary search steps to find the optimal const.")

    #JSMA
    parser.add_argument('--theta', type=float, default=1.0, help='theta')
    parser.add_argument('--gamma', type=float, default=0.1, help="gamma")

    #AA
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--norm', type=str, default='Linf')
    # parser.add_argument('--epsilon', type=float, default=0.15)
    # parser.add_argument('--model', type=str, default='./model_test.pt')
    parser.add_argument('--n_ex', type=int, default=19000)#26400
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./resultsaa')
    # parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--state-path', type=Path, default=None)
    # parser.add_argument('--model', type=str, default="CNN1D")
    # parser.add_argument('--gpu_index', type=str, default='1', help="gpu index to use")
    # parser.add_argument('--dataset', type=str, default='128', help="gpu index to use")
    # parser.add_argument('--name', type=str, default='r8conv1', help='the model ')

#some defense para
    parser.add_argument('--eps', type=float, default=0.3, help='magnitude of random space')
    parser.add_argument('--step_num', type=int, default=40, help='perform how many steps when PGD perturbation')
    parser.add_argument('--step_size', type=float, default=0.03, help='the size of each perturbation')

    # parameters for the NAT Defense
    parser.add_argument('--adv_ratio', type=float, default=0.3,
                        help='the weight of adversarial example when adversarial training')
    parser.add_argument('--clip_min', type=float, default=0.0, help='the min of epsilon allowed')
    parser.add_argument('--clip_max', type=float, default=0.3, help='the max of epsilon allowed')
    parser.add_argument('--eps_mu', type=int, default=0, help='the \mu value of normal distribution for epsilon')
    parser.add_argument('--eps_sigma', type=int, default=50, help='the \sigma value of normal distribution for epsilon')
    parser.add_argument('--lamba1', type=float, default=1.0, help='')
    parser.add_argument('--lamba2', type=float, default=1.0, help='')
    #DD
    parser.add_argument('--initial', type=lambda x: (str(x).lower() == 'true'), default='True',
                        help='True if there exists a pre-trained initial model')
    parser.add_argument('--temp', type=float, default=30.0, help='distillation temperature')




    args = parser.parse_args()
    return args


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()

