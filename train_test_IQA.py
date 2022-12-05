import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import random
import numpy as np
from HyerIQASolver import HyperIQASolver

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(config):
    data_root = '/home/sunnypc/dangxs/datasets/IQA/'
    folder_path = {
        'live': f'{data_root}/LIVE1/data/databaserelease2/',
        'csiq': f'{data_root}/CSIQ/',
        'tid2013': f'{data_root}/tid2013/',

        # warning: 每种数据集的label 范围不一样, 不能混合训练
        'csiq & tid2013': [
            f'{data_root}/CSIQ/',
            f'{data_root}/tid2013/',
        ],

        'live & csiq & tid2013': [
            f'{data_root}/LIVE1/data/databaserelease2/',
            f'{data_root}/CSIQ/',
            f'{data_root}/tid2013/',
        ]

        # 'livec': '/home/ssl/Database/ChallengeDB_release/ChallengeDB_release/',
        # 'koniq-10k': '/home/ssl/Database/koniq-10k/',
        # 'bid': '/home/ssl/Database/BID/',
    }

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),

        'csiq & tid2013':(
            list(range(0, 30)),
            list(range(0, 25)),
        ),

        'live & csiq & tid2013': (
            list(range(0, 29)),
            list(range(0, 30)),
            list(range(0, 25)),
        )

        # 'livec': list(range(0, 1162)),
        # 'koniq-10k': list(range(0, 10073)),
        # 'bid': list(range(0, 586)),
    }
    sel_num = img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i + 1))
        # Randomly select 80% images for training and the rest for testing
        if isinstance(sel_num, list):   # single dataset
            random.shuffle(sel_num)
            train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
            test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        elif isinstance(sel_num, tuple):     # mix dataset
            train_index = []
            test_index = []
            for _sel_num in sel_num:
                random.shuffle(_sel_num)
                _train_index = _sel_num[0:int(round(0.8 * len(_sel_num)))]
                _test_index = _sel_num[int(round(0.8 * len(_sel_num))):len(_sel_num)]
                train_index.append(_train_index)
                test_index.append(_test_index)

        solver = HyperIQASolver(config, folder_path[config.dataset], train_index, test_index)
        srcc_all[i], plcc_all[i] = solver.train()

    # print(srcc_all)
    # print(plcc_all)
    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))

    # return srcc_med, plcc_med


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='tid2013',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013|mix')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25,
                        help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')
    parser.add_argument('--save_path', dest='save_path', type=str, default='./train_result',
                        help='path to save train result')
    parser.add_argument('--resume_ckpt', dest='resume_ckpt', type=str, default='./pretrained/koniq_pretrained.pkl',
                        help='load checkpoint to resume training.')


    config = parser.parse_args()
    main(config)
