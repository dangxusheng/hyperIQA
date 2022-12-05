#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: Jory.d
@contact: 707564875@qq.com
@site:
@software: PyCharm
@file: model_inference.py
@time: 2022/11/30 下午1:04
@desc: 使用模型进行推理, 可用于数据处理
"""

import os, os.path as osp
import glob
import torch
import torchvision
import models
import models_mini
from PIL import Image
import cv2
import numpy as np
import img_da

preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize((512, 384)),
    torchvision.transforms.CenterCrop(size=224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))])


def glob_files_py2(read_path, format=0):
    """
    find all files by format
    :param read_path:
    :param format: 0=pic 1=video
    :return:
    """
    import re
    filelist = []
    pattern = None
    if format == 0:
        pattern = re.compile(r'.(png|jpeg|jpg|bmp|BMP|tif|tiff)$')
    elif format == 1:
        pattern = re.compile(r'.(avi|mp4|h264|264|dav|mkv|wmv)$')
    assert pattern is not None
    for r, dirs, files in os.walk(read_path):
        for file in files:
            if re.search(pattern, file) is not None:
                f = '/'.join([r, file])
                if osp.isfile(f): filelist.append(f)
    return filelist


def get_img_list():
    # root_path = '/home/sunnypc/dangxs/datasets/IQA/tid2013/distorted_images'
    root_path = '/home/sunnypc/dangxs/datasets/IQA/CSIQ/src_imgs'
    # root_path = '/home/sunnypc/dangxs/datasets/IQA/LIVE1/data/databaserelease2/refimgs'

    files = glob_files_py2(root_path)
    sorted(files)
    return files


@torch.no_grad()
def model_infer():
    device = torch.device("cuda:0")

    # big model: resnet50
    # pretrained_path = './pretrained/koniq_pretrained.pkl'
    # pretrained_path = './train_result/20221201_tid2013/epoch_best.pth'
    # assert osp.exists(pretrained_path)
    # model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)

    # # small model: mobilenet_v2
    pretrained_path = './train_result/20221201_tid2013_mobilenetv2/epoch_best.pth'
    assert osp.exists(pretrained_path)
    model_hyper = models_mini.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).to(device)

    model_hyper.train(False)
    model_hyper.eval()
    print('model create is done.')

    # load our pre-trained model on the koniq-10k dataset
    model_hyper.load_state_dict((torch.load(pretrained_path)))
    print('load checkpoint is done.')

    # tmp_savepath = './infer_save/tid2013/distorted_images'
    # tmp_savepath = './infer_save/resnet50/CSIQ/src_imgs/blur'
    # tmp_savepath = './infer_save/mobilenetv2/CSIQ/src_imgs/blur'
    # tmp_savepath = './infer_save/mobilenetv2/CSIQ/src_imgs/noise'
    tmp_savepath = './infer_save/mobilenetv2/CSIQ/src_imgs/gamma'
    # tmp_savepath = './infer_save/resnet50/LIVE1/data/databaserelease2/refimgs/blur'
    # tmp_savepath = './infer_save/mobilenetv2/LIVE1/data/databaserelease2/refimgs/blur'

    img_list = get_img_list()
    assert len(img_list) > 0
    for f in img_list:
        print(f)
        img = cv2.imread(f)
        assert img is not None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print('src_img.shape: ', img_rgb.shape)
        all_distortion_level_imgs = [img_rgb, ]

        # # blur
        # for i in range(3, 20, 6):
        #     # img_x = cv2.GaussianBlur(img_rgb, (i,i), 10)
        #     img_x = cv2.blur(img_rgb,(i,i))
        #     all_distortion_level_imgs.append(img_x)

        # 模型对noise的捕捉不敏感
        # # noise
        # import copy
        # for i in range(1,4):
        #     x = copy.deepcopy(img_rgb)
        #     # img_x = img_da.salt_noise(x, 0.98 * (1-i/10))
        #     img_x = img_da.gaussian_noise(x,10, (i+1)*10)
        #     all_distortion_level_imgs.append(img_x)

        # gamma
        import copy
        for i in range(1,4):
            x = copy.deepcopy(img_rgb)
            img_x = img_da.gamma_transform(x,i+1)
            all_distortion_level_imgs.append(img_x)

        for _img in all_distortion_level_imgs:
            img_pil = Image.fromarray(_img)
            img_i = preprocess(img_pil)[None, ...]
            img_i = torch.as_tensor(img_i, dtype=torch.float32).to(device)

            pred = model_hyper.onnx_forward(img_i)

            # paras = model_hyper.forward(img_i)  # 'paras' contains the network weights conveyed to target network
            # # Building target network
            # model_target = models.TargetNet(paras).cuda()
            # for param in model_target.parameters():
            #     param.requires_grad = False
            # # Quality prediction
            # pred2 = model_target(paras['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
            # print(f'{f}, IQA Score: {pred}  <---> {pred2}')

            iqa_score = float(pred.item())
            cv2.putText(_img, 'IQA: {:.2f}'.format(iqa_score), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 2)
            cv2.resize(_img, (224, 224), dst=_img)

        img = np.concatenate(all_distortion_level_imgs, axis=1)
        cv2.cvtColor(img,cv2.COLOR_RGB2BGR,dst=img)
        _savepath = f'{tmp_savepath}/{osp.basename(f)}'
        os.makedirs(osp.dirname(_savepath), exist_ok=True)
        cv2.imwrite(_savepath, img)

    print('done.')


"""
python model_inference.py
"""

if __name__ == '__main__':
    model_infer()
