#!/usr/bin/env python
# encoding: utf-8
"""
@version: v1.0
@author: Jory.d
@contact: 707564875@qq.com
@site: 
@software: PyCharm
@file: ncnn_basenet.py
@time: 2022/10/28 下午3:28
@desc:  用于ncnn推理的基础类
"""

from abc import ABCMeta, abstractmethod
import os, os.path as osp
import numpy as np
import ncnn
import cv2
import pickle


class NCNNBaseNet(metaclass=ABCMeta):
    CLASSES = ('object1', '__backgound__',)

    MODEL_ROOT = './ncnn_models/repvgg'
    PARAM_PATH = f'{MODEL_ROOT}/epoch_635.deploy.pth.sim-opt-fp16.param'
    BIN_PATH = f'{MODEL_ROOT}/epoch_635.deploy.pth.sim-opt-fp16.bin'

    INPUT_W = 112
    INPUT_H = 112
    INPUT_C = 3
    MEAN = [128., ] * INPUT_C
    STD = [1 / 128., ] * INPUT_C

    def __init__(self):
        self.mean_vals = self.MEAN
        self.norm_vals = self.STD
        self.class_num = len(self.CLASSES)
        self.use_gpu = False
        self.num_threads = 4
        self.init_net()

    def init_net(self):
        assert osp.exists(self.PARAM_PATH)
        assert osp.exists(self.BIN_PATH)

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu

        self.net.load_param(self.PARAM_PATH)
        self.net.load_model(self.BIN_PATH)
        print('self.net.load_param() is done.')
        print('self.net.load_model() is done.')

        input_names = self.net.input_names()
        output_names = self.net.output_names()
        assert len(input_names) > 0 and len(output_names) > 0
        print(f'input_names: {input_names}')
        print(f'output_names: {output_names}')

        self.input_names = input_names
        self.output_names = output_names
        self.class_num = len(self.CLASSES)

    def __del__(self):
        self.net = None
        self.anchors_per_level = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, dim=-1):
        x = np.exp(x)
        sum_value = np.sum(x, axis=dim)
        sum_value = np.expand_dims(sum_value, axis=dim)
        return x / sum_value

    def preprocess(self, img):
        """
        :param img:  opencv_mat
        :return: ncnn:mat
        """
        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = img_rgb
        if self.INPUT_C == 1:
            x = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        mat_in = ncnn.Mat.from_pixels_resize(
            x,
            ncnn.Mat.PixelType.PIXEL_GRAY if self.INPUT_C == 1 else ncnn.Mat.PixelType.PIXEL_RGB,
            img_w,
            img_h,
            self.INPUT_W,
            self.INPUT_H,
        )
        mat_in.substract_mean_normalize(self.MEAN, self.STD)
        return mat_in

    @abstractmethod
    def detect(self, img, thres):
        raise NotImplementedError('must be overwrite.')
        pass


# 目标检测基础类
class NCNNDetectNet(NCNNBaseNet):
    ANCHOR_PATH = f'./epoch_7000_anchor_per_level.pkl'
    OUTPUT_NODES = [
        ['61', '62'],  # stride 32 -- (conf, loc)
    ]

    bbox_means = [0., 0., 0., 0.]
    bbox_stds = [0.1, 0.1, 0.2, 0.2]

    def __init__(self):
        super(NCNNDetectNet, self).__init__()
        self.load_anchor_data()

    def load_anchor_data(self):
        assert osp.exists(self.ANCHOR_PATH)
        f = open(self.ANCHOR_PATH, 'rb')
        # [level1, level2, ...]
        """
        [
            [
                [x1,y1,x2,y2],
                ...
            ],
            [
                [x1,y1,x2,y2],
                ...
            ],
                ...
            ]
        """
        self.anchors_per_level = pickle.load(f)
        f.close()

    def ssd_decode(self, priors, bbox_pred):
        """
        SSD DeltaXYWHBBoxCoder 方式的box解码， 参考自 mmdetection / DeltaXYWHBBoxCoder
        """
        deltas = bbox_pred
        rois_ = np.asarray(priors).reshape(-1, 4)
        assert rois_.shape == bbox_pred.shape, f'{rois_.shape} != {bbox_pred.shape}.'
        means, stds = self.bbox_means, self.bbox_stds
        means, stds = np.array([means]), np.array([stds])
        denorm_deltas = deltas * stds + means
        dxy = denorm_deltas[:, :2]
        dwh = denorm_deltas[:, 2:]

        pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
        pwh = (rois_[:, 2:] - rois_[:, :2])
        dxy_wh = pwh * dxy

        gxy = pxy + dxy_wh
        gwh = pwh * np.exp(dwh)
        x1y1 = gxy - (gwh * 0.5)
        x2y2 = gxy + (gwh * 0.5)
        bboxes = np.concatenate([x1y1, x2y2], axis=-1)

        min_xy = 0
        maxXY = (self.INPUT_W, self.INPUT_H)
        max_xy = np.array([[*maxXY] * 2], np.float32)
        bboxes = np.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = np.where(bboxes > max_xy, max_xy, bboxes)
        return bboxes

    def nms_bboxes(self, boxes, scores, nms_thres=0.6):
        """Suppress non-maximal boxes.

        # Arguments
            boxes: ndarray, boxes of objects.
            scores: ndarray, scores of objects.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1)
            h1 = np.maximum(0.0, yy2 - yy1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
            inds = np.where(ovr <= nms_thres)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def post_process(self, outs, thres=0.5):
        mlvl_bboxes, mlvl_scores, mlvl_labels = outs
        nboxes, nclasses, nscores = [], [], []
        for c in range(self.class_num):
            inds = np.where(mlvl_labels == c)
            b = mlvl_bboxes[inds]
            c = mlvl_labels[inds]
            s = mlvl_scores[inds]

            inds = np.where(s >= thres)[0]
            b = b[inds]
            c = c[inds]
            s = s[inds]
            if b.shape[0] == 0: continue

            keep = self.nms_bboxes(b, s)
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if len(nboxes) > 0:
            bboxes = np.concatenate(nboxes)  # [N,4]
            classes = np.concatenate(nclasses)  # [N,]
            scores = np.concatenate(nscores)  # [N,]
            return bboxes, classes, scores
        else:
            return [], [], []

    def detect(self, img, thres=0.7):
        """
        img: opencv_mat, channel order: RGB
        """
        img_h, img_w = img.shape[:2]
        mat_in = self.preprocess(img)
        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)

        ex.input(self.input_names[0], mat_in)

        cls_score_list = []
        bbox_pred_list = []
        for out_names in self.OUTPUT_NODES:  # stride 从小到大
            assert len(out_names) == 2, '2个featmap, 代表score和location.'
            ret, out_conf_stride_n = ex.extract(out_names[0])
            ret, out_loc_stride_n = ex.extract(out_names[1])
            out_conf_stride_n = np.array(out_conf_stride_n)  # [c,h,w]
            out_loc_stride_n = np.array(out_loc_stride_n)
            # print(out_conf_stride_n.shape)
            # print(out_loc_stride_n.shape)
            # print(out_conf_stride_n[0][0][:10])
            # print(out_loc_stride_n[0][0][:10])
            cls_score_list.append(out_conf_stride_n)
            bbox_pred_list.append(out_loc_stride_n)

        mat_in.release()
        ###########################################################################
        mlvl_priors = self.anchors_per_level
        mlvl_bboxes, mlvl_scores, mlvl_labels = [], [], []
        for level_idx, (cls_score, bbox_pred, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, mlvl_priors)):

            cls_score = np.transpose(cls_score, axes=[1, 2, 0]).reshape(-1, self.class_num)
            bboxes = np.transpose(bbox_pred, axes=[1, 2, 0]).reshape(-1, 4)  #

            if self.class_num == 1:
                cls_score = self.sigmoid(cls_score)
            else:
                cls_score = self.softmax(cls_score, -1)

            bboxes = self.ssd_decode(priors, bboxes)

            valid_mask = np.where(cls_score > thres)  # （行,列）
            labels = valid_mask[1]
            cls_score = cls_score[valid_mask]
            bboxes = bboxes[valid_mask[0]]

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score)
            mlvl_labels.append(labels)

        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)
        mlvl_labels = np.concatenate(mlvl_labels, axis=0)
        # print(mlvl_scores, mlvl_bboxes)

        # scale detect size to src
        scale_factor = np.array([[img_w / self.INPUT_W, img_h / self.INPUT_H, ] * 2])
        mlvl_bboxes *= scale_factor

        result = self.post_process((mlvl_bboxes, mlvl_scores, mlvl_labels), thres)
        return result


# NCNN分割基础类
class NCNNSegNet(NCNNBaseNet):
    OUTPUT_NODES = [
        ['61', '62']  # heatmap
    ]

    def detect(self, img, thres=0.7):
        mat_in = self.preprocess(img)
        ex = self.net.create_extractor()
        ex.set_num_threads(self.num_threads)
        ex.input(self.input_names[0], mat_in)

        outs = []
        for node in self.OUTPUT_NODES:
            ret, out_holes_seg = ex.extract(node)  # [n,k,k]
            out_holes_seg = np.array(out_holes_seg)
            outs.append(out_holes_seg)

        mat_in.release()
        return outs

    @abstractmethod
    def post_process(self, outs, thres):
        raise NotImplementedError('must be overwrite.')
        pass

