import argparse
import os
import pathlib
import random
import shutil
import sys

import numpy as np
import torch
sys.path.append('%s/../' % os.path.dirname(os.path.realpath(__file__)))
from dataset import get_single_db_signal_test_loader
from models.network import define_model
from args import args
from tqdm import tqdm
from utils import load_pretrained_model, save_checkpoint


def main():#args
    # Set the random seed manually for reproducibility.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
    device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
    print("CUDA:", args.gpu_index)
    # device = 'cuda:%d' % args.gpu_index
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # prepare the dataset name, candidate num, dataset location and raw model location
    dataset = args.dataset.upper()
    num = args.number
    raw_model_location = args.location
    print("\nStarting to select {} {} Candidates Example, which are correctly classified by the Raw Model from {}\n".format(num, dataset,
                                                                                                                            raw_model_location))

    # load the raw model and testing dataset
    assert args.dataset == '128' or '512' or '1024' or '3040'
    if dataset == '128' or '512' or '1024' or '3040':
        raw_model = define_model(name=args.model)
        # print(raw_model.state_dict().keys())
        # raw_model.load_state_dict(torch.load(raw_model_location,map_location=f"cuda:{args.gpu_index }"))#["net"]

        checkpoint = torch.load(raw_model_location, map_location='cuda:{}'.format(args.gpu_index))
        # load_pretrained_model(raw_model, checkpoint) # ['net']
        load_pretrained_model(raw_model, checkpoint['net']) #

        # test_loader = get_signal_test_loader(batch_size=1, shuffle=False)
        # test_loader = get_alldb_signal_test_loader(batch_size=1, shuffle=False)
        test_loader = get_single_db_signal_test_loader(batch_size=1, shuffle=False)
        # test_loader = get_upper_minus4db_signal_test_loader(batch_size=1, shuffle=False) # 128攻击测试>=-4

    else:
        print("Data error")
    # get the successfully classified examples

    successful = []
    raw_model.eval()

    test_loader = tqdm(test_loader)
    with torch.no_grad():
        # for i, (img, target) in enumerate(train_loader, start=1):
        for i, (signal, label) in enumerate(test_loader, start=1):
            signal = signal.float().to(device)
            label = label.float().to(device)
            output = raw_model(signal)
            _, predicted = torch.max(output.data, 1)

            if predicted == label:
                _, least_likely_class = torch.min(output.data, 1)
                successful.append([signal, label, least_likely_class])

    print("#successful:",len(successful),' ',"ACC:","%.2f" % (100*len(successful)/len(test_loader)),'%')
    print(len(successful), '&&&', len(test_loader))
    candidates = random.sample(successful, num)
    # candidates = successful

    candidate_signal = []
    candidate_labels = []
    candidates_llc = []
    candidate_targets = []

    if args.dataset == '128':
        num_classes = 11
    elif args.dataset == '512':
        num_classes = 12
    elif args.dataset == '1024':
        num_classes = 24
    elif args.dataset == '3040':
        num_classes = 106

    for index in range(len(candidates)):
        signal = candidates[index][0].cpu().numpy()
        signal = np.squeeze(signal, axis=0)
        candidate_signal.append(signal)

        label = int(candidates[index][1].cpu().numpy()[0])
        llc = candidates[index][2].cpu().numpy()[0]

        # selection for the targeted label
        classes = [i for i in range(num_classes)]
        classes.remove(label)
        target = random.sample(classes, 1)[0]

        one_hot_label = [0 for i in range(num_classes)]
        one_hot_label[label] = 1

        one_hot_llc = [0 for i in range(num_classes)]
        one_hot_llc[llc] = 1

        one_hot_target = [0 for i in range(num_classes)]
        one_hot_target[target] = 1

        candidate_labels.append(one_hot_label)
        candidates_llc.append(one_hot_llc)
        candidate_targets.append(one_hot_target)

    candidate_signal = np.array(candidate_signal)
    candidate_labels = np.array(candidate_labels)
    candidates_llc = np.array(candidates_llc)
    candidate_targets = np.array(candidate_targets)

    clean_data_location='./'

    if args.model not in os.listdir(clean_data_location):
        os.mkdir(clean_data_location + args.model +'/')
    elif args.dataset not in os.listdir(clean_data_location + args.model + '/'):
        os.mkdir(clean_data_location + args.model + '/' + args.dataset + '/' )
    else:
        # shutil.rmtree('{}'.format(dataset))
        # os.mkdir(clean_data_location)
        print()
    if not os.path.exists('{}{}/{}'.format(clean_data_location, args.model, args.dataset)):
        os.mkdir('{}{}/{}'.format(clean_data_location, args.model, args.dataset))
    np.save('{}{}/{}/{}_inputs.npy'.format(clean_data_location, args.model, args.dataset, args.dataset), candidate_signal)
    np.save('{}{}/{}/{}_labels.npy'.format(clean_data_location, args.model, args.dataset, args.dataset), candidate_labels)
    np.save('{}{}/{}/{}_llc.npy'.format(clean_data_location, args.model, args.dataset, args.dataset), candidates_llc)
    np.save('{}{}/{}/{}_targets.npy'.format(clean_data_location, args.model, args.dataset, args.dataset), candidate_targets)

    write_result_to_csv(
        model=args.model,
        dataset=args.dataset,
        snr=args.db,
        note=args.note,
        number=args.number,
        acc="%.2f" % (100*len(successful)/len(test_loader))
    )

def write_result_to_csv(**kwargs):
    results = pathlib.Path("./") / "results.csv"

    if not results.exists():
        results.write_text(
            "MODEL, "
            "DATA, "
            "SNR, "
            "NOTE, "
            "#DATA, "
            "ACC\n "
        )

    with open(results, "a+") as f:  # a+附加读写方式打开
        f.write(
            ("{model}, "
             "{dataset}, "
             "{snr}, "
             "{note}, "
             "{number}, "
             "{acc}\n"
             ).format(**kwargs)
        )


if __name__ == '__main__':
    main()

