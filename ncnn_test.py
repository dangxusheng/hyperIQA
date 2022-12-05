#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: Jory.d
@contact: 707564875@qq.com
@site: 
@software: PyCharm
@file: ncnn_test.py
@time: 2022/11/16 上午10:51
@desc: 以验证, 和pytorch\onnx的结果基本一致
"""

"""
python ncnn_test.py

platform： linux_x86_64
inference time:
    mode: fp16
        forward used:       44 ms
        postprocess used:   5 ms
    mode: fp32
        forward used:       60 ms
        postprocess used:   6 ms


"""

import cv2
import numpy as np
from ncnn_basenet import NCNNBaseNet
import time

class NCNN_IQANet(NCNNBaseNet):
    CLASSES = ('IQA_SCORE',)

    # MODEL_ROOT = './pretrained'
    # # PARAM_PATH = f'{MODEL_ROOT}/koniq_pretrained.onnx.opt.param'
    # # BIN_PATH = f'{MODEL_ROOT}/koniq_pretrained.onnx.opt.bin'
    # PARAM_PATH = f'{MODEL_ROOT}/koniq_pretrained.onnx.opt.fp16.param'
    # BIN_PATH = f'{MODEL_ROOT}/koniq_pretrained.onnx.opt.fp16.bin'
    # OUTPUT_NODES = ['573',
    #                 '584', '591',
    #                 '594', '601',
    #                 '604', '611',
    #                 '614', '621',
    #                 '628', '635'
    #                 ]

    MODEL_ROOT = './train_result/20221201_tid2013_mobilenetv2'
    PARAM_PATH = f'{MODEL_ROOT}/epoch_best.pth.onnx.opt.param'
    BIN_PATH = f'{MODEL_ROOT}/epoch_best.pth.onnx.opt.bin'
    # PARAM_PATH = f'{MODEL_ROOT}/epoch_best.pth.onnx.opt.fp16.param'
    # BIN_PATH = f'{MODEL_ROOT}/epoch_best.pth.onnx.opt.fp16.bin'
    OUTPUT_NODES = ['614',
                    '625', '632',
                    '635', '642',
                    '645', '652',
                    '655', '662',
                    '669', '676'
                    ]


    INPUT_W = 224
    INPUT_H = 224
    INPUT_C = 3
    MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
    STD = (1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255))


    def detect(self, img, thres=0.7):
        mat_in = self.preprocess(img)
        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)
        ex.input(self.input_names[0], mat_in)

        s = time.time()
        outs = []
        for node in self.OUTPUT_NODES:
            assert node in self.output_names, f'{node} not in {self.output_names}'
            ret, out = ex.extract(node)  # [n,k,k]
            out = np.asarray(out)
            # print(out.shape)
            outs.append(out)
        print(f'cnn forward() elasped : {time.time() - s} s', )
        mat_in.release()

        out = self.postprocess(outs)

        return out

    def postprocess(self, outs):
        target_in_vec = outs[0]
        """
        (224, 1, 1)
            (112, 224, 1)
            (112,)
            (56, 112, 1)
            (56,)
            (28, 56, 1)
            (28,)
            (14, 28, 1)
            (14,)
            (1, 14, 1)
            (1,)
        """

        x = target_in_vec  # [224,1,1]
        for i in range(1, 11, 2):
            # x.shape: [h,w]
            x = np.reshape(x, [x.shape[0], -1])  # [c,1]
            w, b = outs[i], outs[i + 1]
            # w.shape: [c,h,1]
            # b.shape: [c,]
            # print('x: ', x.shape)
            # print('w: ', w.shape)
            # print('b: ', b.shape)
            # 相当于点乘求和
            xx = [np.sum(x * w[c]) + b[c] for c in range(w.shape[0])]
            x = np.asarray(xx)[..., None]
            # print('w*x+b: ', x.shape)
            if i < 9:  x = self.sigmoid(x)
        x = x.reshape(-1)   #[1,]
        return x


if __name__ == "__main__":
    print('hello')
    # x = cv2.imread('./1600.BLUR.5.png')
    x = cv2.imread("./data/D_03.jpg")
    # x = np.random.randint(0, 255, [224, 224, 3], dtype=np.uint8)
    m = NCNN_IQANet()

    s = time.time()
    iqa = m.detect(x)
    print('IQA: ', iqa)
    print(f'detect() elasped : {time.time() - s} s', )
