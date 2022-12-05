import torch
from scipy import stats
import numpy as np
import models
import models_mini
import data_loader
import os


class HyperIQASolver(object):
    """Solver for training and testing hyperIQA"""

    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.save_path = config.save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.resume_ckpt = config.resume_ckpt
        # assert os.path.exists(self.resume_ckpt), f'{self.resume_ckpt} is not existed.'

        # resnet50 有官方预训练模型, 很快收敛
        # self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7, backbone='resnet50').cuda()
        # resent18 过拟合, 估计可没有预训练有很大关系
        # self.model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7, backbone='resnet18').cuda()
        # 替换mobilenetV2
        self.model_hyper = models_mini.HyperNet(16, 112, 224, 112, 56, 28, 14, 7, backbone='mobilenet_v2').cuda()
        self.model_hyper.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()
        self.loss_weight = 1.5

        pretrained_path = self.resume_ckpt
        # # full param, load our pre-trained model on the koniq-10k dataset
        if os.path.exists(pretrained_path):
            self.model_hyper.load_state_dict((torch.load(pretrained_path)), strict=True)
        #     # optical param load
        #     save_model = torch.load(pretrained_path)
        #     model_dict = self.model_hyper.state_dict()
        #     state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys() and model_dict[k].shape == v.shape}
        #     # from pprint import pprint
        #     # pprint(state_dict.keys())
        #     self.model_hyper.load_state_dict(state_dict, strict=True)
            print('load checkpoint is done.')

        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)
        self.current_lr = self.lr

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size,
                                              config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num,
                                             istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\t\tTrain_Loss\t\tTrain_SRCC\t\tTrain_PLCC\t\tTest_SRCC\t\tTest_PLCC\t\tLearning Rate')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            n = len(self.train_data)
            for step, (img, label) in enumerate(self.train_data):
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())

                self.solver.zero_grad()

                # Generate weights for target network
                paras = self.model_hyper(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net

                # pred = torch.sigmoid(pred)

                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                loss *= self.loss_weight

                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

                if step % 50 == 0:
                    train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
                    train_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
                    print('%d/%d:\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t--\t\t--\t\t%9.9f' %
                          (step, n, sum(epoch_loss) / len(epoch_loss), train_srcc, train_plcc, self.current_lr))

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            train_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

            if t % 2 == 0:
                # save
                _ckpt_path = f'./{self.save_path}/epoch_%d.pth' % t
                os.makedirs(os.path.dirname(_ckpt_path), exist_ok=True)
                torch.save(self.model_hyper.state_dict(), _ckpt_path)

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                # save
                _ckpt_path = f'./{self.save_path}/epoch_best.pth'
                os.makedirs(os.path.dirname(_ckpt_path), exist_ok=True)
                torch.save(self.model_hyper.state_dict(), _ckpt_path)

            print('epoch:%d\t\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t--' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, train_plcc, test_srcc, test_plcc))

            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
            self.current_lr = lr

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        self.model_hyper.train(False)
        pred_scores = []
        gt_scores = []

        for img, label in data:
            # Data.
            img = torch.tensor(img.cuda())
            label = torch.tensor(label.cuda())

            paras = self.model_hyper(img)
            model_target = models.TargetNet(paras).cuda()
            model_target.train(False)
            pred = model_target(paras['target_in_vec'])
            # pred = torch.sigmoid(pred)

            pred_scores.append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_hyper.train(True)
        return test_srcc, test_plcc
