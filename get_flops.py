#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import numpy as np
import models
import models_mini
import torch

"""
计算模型的FLOPS
python get_flops.py
"""

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')

parser = argparse.ArgumentParser(description='Train a detector')
args = parser.parse_args()
args.shape = (3, 224, 224)
# args.shape = (3, 112, 112)
args.size_divisor = 32

# # Flops: 4.32 G
# model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7, backbone='resnet50')

# # Flops: 3.73 G
# model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7, backbone='resnet34')

# Flops: 1.87 G
# model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7, backbone='resnet18')
model = models_mini.HyperNet(16, 112, 224, 112, 56, 28, 14, 7, backbone='mobilenet_v2')

print("\t Successfully initialized model")

# # optical param load
# save_model = torch.load('./pretrained/koniq_pretrained.pkl')
# model_dict = model.state_dict()
# state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys() and model_dict[k].shape == v.shape}
# model_dict.update(state_dict)
# model.load_state_dict(model_dict, strict=True)


def main():
    c, h, w = tuple(args.shape)
    ori_shape = (c, h, w)
    divisor = args.size_divisor
    if divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor

    input_shape = (c, h, w)
    model.eval()
    model.forward = model.onnx_forward

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30

    if divisor > 0 and \
            input_shape != ori_shape:
        print(f'{split_line}\nUse size divisor set input shape '
              f'from {ori_shape} to {input_shape}\n')
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
