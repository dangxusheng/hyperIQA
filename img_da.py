#!/usr/bin/env python
# -*- coding:utf-8-*-
# file: {NAME}.py
# @author: jory.d
# @contact: dangxusheng163@163.com
# @time: 2021/08/01 18:20
# @desc: 

"""
基于opencv的样本数据增强
"""

import math
import random
import cv2
import numpy as np
import scipy
from skimage import exposure, util, color
from PIL import Image, ImageEnhance
import copy


def random_noise(img, rand_range=(3, 15)):
    """
    随机+-噪声
    :param img:
    :param rand_range: (min, max)
    :return:
    """
    img = np.asarray(img, np.float)
    sigma = random.randint(*rand_range)
    nosie = np.random.normal(0, sigma, size=img.shape)
    if np.random.rand() < 0.5:
        nosie *= -1
    img += nosie
    return np.clip(img, 0, 255)


def noise_optional(img, type='gaussian'):
    from skimage import util
    assert type in ('gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle')
    return util.random_noise(img, type)


def salt_noise(img, snr=0.98):
    """
    椒盐噪声,
    假设一张图像的宽x高 = 10x10 ，共计100个像素，想让其中20个像素点变为噪声，其余80个像素点保留原值，则这里定义的SNR=80/100 = 0.8
    :param img:
    :param snr: 信噪比（Signal-Noise Rate, SNR）, 越大保留原图像越多
    :return:
    """
    src_shape_len = len(img.shape)
    if src_shape_len < 3:
        img = np.expand_dims(img, axis=-1)
    h, w, c = img.shape
    SNR = snr
    mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    if c > 1:
        mask = np.repeat(mask, c, axis=-1)  # 按channel 复制到 与img具有相同的shape
    img[mask == 1] = 255  # 盐噪声
    img[mask == 2] = 0  # 椒噪声
    del mask
    if src_shape_len < 3:
        img = np.squeeze(img, axis=-1)
    return img


def add_salt_and_pepper(img):
    """ Add salt and pepper noise to an image. """
    noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.randu(noise, 0, 255)
    black = noise < 30
    white = noise > 225
    img[white > 0] = 255
    img[black > 0] = 0
    cv2.blur(img, (5, 5), img)
    return img


def gaussian_noise(img, mean=0., sigma=1.):
    """
    高斯噪声
    :param img:
    :param mean:
    :param sigma:
    :return:
    """
    src_shape_len = len(img.shape)
    if src_shape_len < 3:
        img = np.expand_dims(img, axis=-1)
    h, w, c = img.shape
    img = img.astype('float64')
    noise_mat = np.random.normal(mean, sigma, size=(h, w, c))
    img += noise_mat
    img = np.clip(img, 0, 255)
    if src_shape_len < 3:
        img = np.squeeze(img, axis=-1)

    return img.astype('uint8')


def addWeight(src1, alpha, src2, beta, gamma):
    """
    两幅图像加权叠加
    dst = src1 * alpha + src2 * beta + gamma
    :param src1:
    :param alpha:
    :param src2:
    :param beta:
    :param gamma:
    :return:
    """
    assert src1.shap == src2.shape
    return cv2.addWeighted(src1, alpha, src2, beta, gamma)


def usm_sharp(img, gamma=0.):
    """
    USM锐化增强算法可以去除一些细小的干扰细节和图像噪声，比一般直接使用卷积锐化算子得到的图像更可靠。
        output = 原图像−w∗高斯滤波(原图像)/(1−w)
	    其中w为上面所述的系数，取值范围为0.1~0.9，一般取0.6。
    :param img:
    :param gamma:
    :return:
    """
    blur = cv2.GaussianBlur(img, (0, 0), 25)
    img_sharp = cv2.addWeighted(img, 1.5, blur, -0.3, gamma)
    return img_sharp


def adjust_contrast_bright(img, contrast=1.2, brightness=100):
    """
    调整亮度与对比度
    dst = img * contrast + brightness
    :param img:
    :param contrast: 对比度   越大越亮
    :param brightness: 亮度  0~100
    :return:
    """
    # 像素值会超过0-255， 因此需要截断
    return np.uint8(np.clip((contrast * img + brightness), 0, 255))


def random_saturation(image, alpha=0.1):
    """
    随机饱和度
    :param image:
    :param alpha: 越接近0,　代表越接近原图
    :return:
    """
    assert len(image.shape) >= 3

    # alpha = np.random.rand()
    image = image.astype(np.float32) / 255.
    hlsimg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # 饱和度
    hlsimg[:, :, 2] = (1.0 + alpha) * hlsimg[:, :, 2]
    hlsimg[:, :, 2][hlsimg[:, :, 2] > 1] = 1

    image = cv2.cvtColor(hlsimg, cv2.COLOR_HLS2BGR)
    return image


def random_hue(image, alpha=1.5):
    """
    图像色度变化
    :param image:
    :param alpha:　1.0代表原图
    :return:
    """
    assert len(image.shape) >= 3

    pil_image = Image.fromarray(image)
    enh_col = ImageEnhance.Color(pil_image)
    pil_image = enh_col.enhance(alpha)
    image = np.asarray(pil_image)
    image = np.clip(image, 0, 255)
    return image


def random_contrast(image, alpha, beta=0):
    """
    随机对比度变换: y = alpha * x + beta
    :param image:
    :param alpha:
    :param beta:
    :return:
    """
    assert 0.4 <= alpha <= 2.
    assert len(image.shape) >= 3
    image = image.astype(np.float32)
    image = image * alpha + beta
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image


def random_brightness(image, alpha=0.1):
    """
    随机亮度
    :param image:
    :param alpha: 越接近0,　代表越接近原图
    :return:
    """
    assert len(image.shape) >= 3
    # alpha = np.random.rand()
    image = image.astype(np.float32) / 255.
    hlsimg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hlsimg[:, :, 1] = (1.0 + alpha) * hlsimg[:, :, 1]
    hlsimg[:, :, 1][hlsimg[:, :, 1] > 1] = 1
    image = cv2.cvtColor(hlsimg, cv2.COLOR_HLS2BGR)
    return image


def random_flip(img, mode=1):
    """
    随机翻转
    :param mode:
    :param img:
    :param model: 1=水平翻转 / 0=垂直 / -1=水平垂直
    :return:
    """
    assert mode in (0, 1, -1), "mode is not right"
    flip = np.random.choice(2) * 2 - 1  # -1 / 1
    if mode == 1:
        img = img[:, ::flip, ...]
    elif mode == 0:
        img = img[::flip, ...]
    elif mode == -1:
        img = img[::flip, ::flip, ...]

    return img


def random_flip_with_bboxes(img, bboxes=[], mode=1):
    """
    :param img:
    :param bboxes: [n,4], (x1,y1,x2,y2)
    :param mode:
    :return:
    """
    assert mode in (0, 1, -1), "mode is not right"
    flip = np.random.choice(2) * 2 - 1  # -1 / 1
    bboxes = np.asarray(bboxes)
    has_box = bboxes.shape[0] > 0
    bboxes_flip = bboxes.copy()
    if has_box:
        assert bboxes.shape[1] == 4

    h, w = img.shape[:2]
    if mode == 1:  # Hflip
        img = img[:, ::flip, ...]
        if has_box:
            x1, x2 = bboxes_flip[:, 0].copy(), bboxes_flip[:, 2].copy()
            bboxes_flip[:, 0] = w - x2
            bboxes_flip[:, 2] = w - x1

    elif mode == 0:  # Vflip
        img = img[::flip, ...]
        if has_box:
            y1, y2 = bboxes_flip[:, 1].copy(), bboxes_flip[:, 3].copy()
            bboxes_flip[:, 1] = h - y2
            bboxes_flip[:, 3] = h - y1
    elif mode == -1:
        img = img[::flip, ::flip, ...]
        if has_box:
            x1, x2 = bboxes_flip[:, 0].copy(), bboxes_flip[:, 2].copy()
            y1, y2 = bboxes_flip[:, 1].copy(), bboxes_flip[:, 3].copy()
            bboxes_flip[:, 0] = w - x2
            bboxes_flip[:, 2] = w - x1
            bboxes_flip[:, 1] = h - y2
            bboxes_flip[:, 3] = h - y1

    bboxes_flip = list(bboxes_flip)
    return img, bboxes_flip


def cv_flip(img, mode=1):
    """
    翻转
    :param img:
    :param mode: 1=水平翻转 / 0=垂直 / -1=水平垂直
    :return:
    """
    assert mode in (0, 1, -1), "mode is not right"
    return cv2.flip(img, flipCode=mode)


def rotate(img, angle, scale=1.0):
    """
    旋转
    :param img:
    :param angle: 旋转角度， >0 表示逆时针，
    :param scale:
    :return:
    """
    height, width = img.shape[:2]  # 获取图像的高和宽
    center = (width / 2, height / 2)  # 取图像的中点

    M = cv2.getRotationMatrix2D(center, angle, scale)  # 获得图像绕着某一点的旋转矩阵
    # cv2.warpAffine()的第二个参数是变换矩阵,第三个参数是输出图像的大小
    rotated = cv2.warpAffine(img, M, (width, height))
    return rotated


def rotate_with_bboxes(img, bboxes, angle, scale=1.0):
    """
    带gt_boxes的图像旋转
    :param img:
    :param bboxes: [b, 4],  format: x1y1x2y2
    :param angle: 旋转角度， >0 表示逆时针，
    :param scale:
    :return:
    """
    height, width = img.shape[:2]  # 获取图像的高和宽
    center = (width / 2, height / 2)  # 取图像的中点

    M = cv2.getRotationMatrix2D(center, angle, scale)  # 获得图像绕着某一点的旋转矩阵
    # cv2.warpAffine()的第二个参数是变换矩阵,第三个参数是输出图像的大小
    rotated_img = cv2.warpAffine(img, M, (width, height))
    height, width = rotated_img.shape[:2]
    M_t = np.transpose(M)
    rotated_boxes = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        pts = np.array([
            [x1, y1],
            [x2, y1],
            [x1, y2],
            [x2, y2],
        ])
        pts = np.matmul(pts, M_t[:2, :]) + M_t[2, :2]
        x1 = max(0, np.min(pts[:, 0]))
        y1 = max(0, np.min(pts[:, 1]))
        x2 = min(width - 1, np.max(pts[:, 0]))
        y2 = min(height - 1, np.max(pts[:, 1]))
        x1, y1, x2, y2 = list(map(int, [x1, y1, x2, y2]))
        rotated_boxes.append([x1, y1, x2, y2])

    return rotated_img, rotated_boxes


def random_rotate(img, angle_range=(-10, 10)):
    """
    随机旋转
    :param img:
    :param angle_range:  旋转角度范围 (min,max)   >0 表示逆时针，
    :return:
    """
    height, width = img.shape[:2]  # 获取图像的高和宽
    center = (width / 2, height / 2)  # 取图像的中点
    angle = random.randrange(*angle_range, 1)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 获得图像绕着某一点的旋转矩阵
    # cv2.warpAffine()的第二个参数是变换矩阵,第三个参数是输出图像的大小
    rotated = cv2.warpAffine(img, M, (width, height))
    return rotated


def random_filter(img, ksize=(5, 5)):
    """
    随机滤波
    https://blog.csdn.net/qq_27261889/article/details/80822270
    :param img:
    :param ksize:
    :return:
    """
    assert ksize[0] % 2 != 0
    blur_types = ['gaussian', 'median', 'bilateral', 'mean', 'box']
    assert len(blur_types) > 0
    blur_index = random.choice([i for i in range(len(blur_types))])
    if blur_index == 0:  # 高斯模糊, 比均值滤波更平滑，边界保留更加好
        img_blur = cv2.GaussianBlur(img, ksize, 1)
    elif blur_index == 1:  # 中值滤波, 在边界保存方面好于均值滤波，但在模板变大的时候会存在一些边界的模糊。对于椒盐噪声有效
        img_blur = cv2.medianBlur(img, ksize[0])
    elif blur_index == 2:  # 双边滤波, 非线性滤波，保留较多的高频信息，不能干净的过滤高频噪声，对于低频滤波较好，不能去除脉冲噪声
        img_blur = cv2.bilateralFilter(img, 9, 75, 75)
    elif blur_index == 3:  # 均值滤波, 在去噪的同时去除了很多细节部分，将图像变得模糊,
        img_blur = cv2.blur(img, ksize)
    elif blur_index == 4:  # 盒滤波器: 对于椒盐噪声有效
        img_blur = cv2.boxFilter(img, -1, ksize)

    return img_blur


def gaussian_filter(img, ks=(7, 7), stdev=1.5):
    """
    高斯模糊, 可以对图像进行平滑处理，去除尖锐噪声
    :param img:
    :param ks:  卷积核
    :param stdev: 标准差
    :return:
    """
    return cv2.GaussianBlur(img, ks, stdev)


def equalize_hist(img):
    """
    全局直方图均衡化
    :param img:
    :return:
    """
    need_to_gray = len(img.shape) == 3 and img.shape[-1] == 3
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if need_to_gray else img
    hist = cv2.equalizeHist(gray)
    rgb = cv2.cvtColor(hist, cv2.COLOR_GRAY2RGB) if need_to_gray else hist
    return rgb


def equalize_hist_AHE(img):
    """
    自适应直方图均衡化   Adaptive histgram equalization/AHE
    :param img:
    :return:
    """
    need_to_gray = len(img.shape) == 3 and img.shape[-1] == 3
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if need_to_gray else img
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
    hist = clahe.apply(gray)
    rgb = cv2.cvtColor(hist, cv2.COLOR_GRAY2RGB) if need_to_gray else hist
    return rgb


def homomorphic_filter(img):
    """
    同态滤波： 去除乘性噪声，增强对比度和标准化亮度，
    :param img:
    :return:
    """
    is_color = len(img.shape) > 2
    if is_color:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    float_img = img.astype('float32')
    log_img = cv2.log(float_img + 1)
    h, w = log_img.shape[:2]
    # 高通滤波， 使滤波器的大小（M，N）为偶数来加速FFT计算
    M, N = h, w  # cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)
    if M % 2 != 0: M += 1
    if N % 2 != 0: N += 1
    pad_img = cv2.copyMakeBorder(log_img, 0, M - h, 0, N - w, cv2.BORDER_CONSTANT)
    h, w = pad_img.shape[:2]
    # 傅里叶变换
    fft_img = cv2.dct(pad_img)
    h1, w1 = fft_img.shape[:2]
    sigma = 3
    # 构造一个高斯频域高通滤波器
    gauss_filter = np.zeros([h1, w1], dtype=np.float32)
    for i in range(h1):
        for j in range(w1):
            pixel_dis = math.sqrt(i * i + j * j)
            coeff = 1.0 - math.exp(-pixel_dis * pixel_dis / (2 * sigma * sigma))
            gauss_filter[i, j] = coeff

    fft_img *= gauss_filter
    i_fft_img = cv2.idct(fft_img, flags=cv2.DFT_INVERSE)
    i_fft_img = cv2.exp(i_fft_img - 1) * 255
    i_fft_img = i_fft_img.astype('uint8')
    del gauss_filter, float_img, log_img, fft_img, pad_img

    if is_color:
        i_fft_img = cv2.cvtColor(i_fft_img, cv2.COLOR_GRAY2RGB)
    return i_fft_img


def random_crop(img, scale=[0.8, 1.0], ratio=[3. / 4., 4. / 3.], resize_w=100, resize_h=100):
    """
    随机裁剪
    :param img:
    :param scale: 缩放
    :param ratio:
    :param resize_w:
    :param resize_h:
    :return:
    """
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio
    src_h, src_w = img.shape[:2]

    bound = min((float(src_w) / src_h) / (w ** 2),
                (float(src_h) / src_w) / (h ** 2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = src_h * src_w * np.random.uniform(scale_min,
                                                    scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, src_w - w + 1)
    j = np.random.randint(0, src_h - h + 1)

    img = img[j:j + h, i:i + w]
    img = cv2.resize(img, (resize_w, resize_h))
    return img


def rule_crop(img, box_ratio=(3. / 4, 3. / 4), location_type='LT', resize_w=100, resize_h=100):
    """
    按照一定规则进行裁剪, 直接在原图尺寸上操作，不对原图进行
    :param img:
    :param box_ratio: 剪切的 比例：  （宽度上的比例， 高度上的比例）
    :param location_type: 具体在=哪个位置： 以下其中一个：
            LR : 左上角
            RT : 右上角
            LB : 左下角
            RB : 右下角
            CC : 中心
    :param resize_w: 输出图的width
    :param resize_h: 输出图的height
    :return:
    """
    assert location_type in ('LT', 'RT', 'LB', 'RB', 'CC'), 'must have a location .'
    is_gray = False
    if len(img.shape) == 3:
        h, w, c = img.shape
    elif len(img.shape) == 2:
        h, w = img.shape
        is_gray = True

    crop_w, crop_h = int(w * box_ratio[0]), int(h * box_ratio[1])
    crop_img = np.zeros([10, 10])
    if location_type == 'LT':
        crop_img = img[:crop_h, :crop_w, :] if not is_gray else img[:crop_h, :crop_w]
    elif location_type == 'RT':
        crop_img = img[:crop_h:, w - crop_w:, :] if not is_gray else img[:crop_h:, w - crop_w:]
    elif location_type == 'LB':
        crop_img = img[h - crop_h:, :crop_w, :] if not is_gray else img[h - crop_h:, :crop_w]
    elif location_type == 'RB':
        crop_img = img[h - crop_h:, w - crop_w:, :] if not is_gray else img[h - crop_h:, w - crop_w:]
    elif location_type == 'CC':
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        crop_img = img[start_h:start_h + crop_h, start_w:start_w + crop_w, :] if not is_gray else img[
                                                                                                  start_h:start_h + crop_h,
                                                                                                  start_w:start_w + crop_w]

    resize = cv2.resize(crop_img, (resize_w, resize_h))
    return resize


def shift(img, x_offset, y_offset):
    """
    偏移，向右 向下
    :param img:
    :param x_offset:  >0表示向右偏移px, <0表示向左
    :param y_offset:  >0表示向下偏移px, <0表示向上
    :return:
    """
    h, w = img.shape[:2]
    M = np.array([[1, 0, x_offset], [0, 1, y_offset]], dtype=np.float)
    return cv2.warpAffine(img, M, (w, h))


def mixup(batch_x, batch_y, alpha):
    """
    mixup: 一种图像混合的数据增强策略
    alpha ~ Beta(alpha,alpha), [0,1]
    x = alpha * x1 + (1-alpha) * x2
    y = alpha * y1 + (1-alpha) * y2
    :param batch_x:
    :param batch_y:
    :param alpha:
    :return: mixed inputs, pairs of targets, and lambda
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = batch_x.shape[0]
    # index = torch.randperm(batch_size)
    index = [i for i in range(batch_size)]
    random.shuffle(index)
    mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
    y_a, y_b = batch_y, batch_y[index]

    # pred = model(mixed_x)
    # loss = lam * criterion(pred,y_a) + (1-lam)*criterion(pred,y_a)
    # acc = lam * criterion(pred,y_a)[0] + (1-lam)*criterion(pred,y_a)[0]

    return mixed_x, y_a, y_b, lam


def gamma_transform(img, gamma=1.0):
    """
    https://blog.csdn.net/zfjBIT/article/details/85113946
    伽马变换就是用来图像增强，其提升了暗部细节，简单来说就是通过非线性变换，
    让图像从暴光强度的线性响应变得更接近人眼感受的响应，即将漂白（相机曝光）或过暗（曝光不足）的图片，进行矫正
    :param img:
    :param gamma: 需要自己先根据样本来设定,
        # gamma = random.random() * random.choice([0.5, 1, 3, 5])
        >1, 变暗
        <1, 漂白
    :return:
    """
    assert 0 < gamma < 25.
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    _mean = np.mean(img)
    if _mean > 220:
        gamma = max(1, gamma)
    elif _mean < 20:
        gamma = min(1, gamma)
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    return cv2.LUT(img, gamma_table)


def cut_out(img, n_holes, hole_size=(10, 10)):
    """
    在图片上随机抠黑洞.
    :param img: np.array
    :param n_holes: 黑洞的数量
    :param hole_size: 黑洞的尺寸
    :return:
    """
    h, w, c = img.shape
    hole_w, hole_h = hole_size
    mask = np.ones_like(img, dtype=np.float32)
    for n in range(n_holes):
        x, y = list(map(np.random.randint, [w, h]))
        y1 = np.clip(y - hole_h // 2, 0, h)
        y2 = np.clip(y + hole_h // 2, 0, h)
        x1 = np.clip(x - hole_w // 2, 0, w)
        x2 = np.clip(x + hole_w // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.

    return (img * mask).astype('uint8')


def erase(img, min_area=0.02, max_area=0.4, aspect_ratio=0.3, values=[100, 100, 100]):
    """
    在图片上按照参数随机擦除
    :param img: np.array
    :param min_area: min erasing area, [0,1]
    :param max_area: max erasing area, [0,1]
    :param aspect_ratio: min aspect ratio, [0,1]
    :param values:  erasing value
    :return:
    """
    assert 0 < aspect_ratio and 0 < min_area < 1 and 0 < max_area < 1
    h, w = img.shape[:2]
    isColor = len(img.shape) == 3 and img.shape[2] == 3 and len(values) == 3
    area = h * w
    target_area = random.uniform(min_area, max_area) * area
    aspect_ratio = random.uniform(aspect_ratio, 1 / aspect_ratio)
    h1 = int(round(math.sqrt(target_area * aspect_ratio)))
    w1 = int(round(math.sqrt(target_area / aspect_ratio)))
    if w1 < w and h1 < h:
        x1 = random.randint(0, h - h1)
        y1 = random.randint(0, w - w1)
        if isColor:
            for j in range(3):
                img[x1:x1 + h1, y1:y1 + w, j] = values[j]
        else:
            img[x1:x1 + h1, y1:y1 + w] = values[0]

    return img


def color_reverse(img):
    """
    图片颜色翻转, x = 255-x
    :param img:
    :return:
    """
    assert isinstance(img, np.ndarray)
    return np.clip(255 - img, 0, 255)


def gen_heatmap(size=50, sigma=5):
    kernel = cv2.getGaussianKernel(size, sigma)
    kernel = kernel * kernel.T
    kernel = kernel / np.max(kernel)
    print((kernel != 0).sum())
    heatmap = kernel * 255
    heatmap = heatmap.astype(np.uint8)
    print(heatmap.shape)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    cv2.imshow('1', heatmap)
    cv2.waitKey(0)


def gen_heatmap2(img, pts, sigma=5):
    h, w = img.shape[:2]
    black_img = np.zeros([h, w])
    for p in pts:
        # p: [x,y]
        size = sigma * 10
        if size % 2 != 0: size += 1
        kernel = cv2.getGaussianKernel(size, sigma)
        kernel = kernel * kernel.T
        kernel = kernel / np.max(kernel)
        x, y = list(map(int, p))
        startY = abs(y - size // 2) if y - size // 2 < 0 else 0
        startX = abs(x - size // 2) if x - size // 2 < 0 else 0
        endY = abs(h - y + size // 2) if y + size // 2 > h else None
        endX = abs(w - x + size // 2) if x + size // 2 > w else None
        kernel = kernel[startY:endY, startX:endX, ...]
        black_img[
        max(0, y - size // 2):min(h, y + size // 2),
        max(0, x - size // 2):min(w, x + size // 2), ...] = kernel

    heatmap = black_img * 255
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    for p in pts:
        x, y = list(map(int, p))
        cv2.circle(heatmap, (x, y), 1, (0, 0, 0), 2)
    cv2.imshow('1', heatmap)
    cv2.waitKey(0)


def unit_test_above():
    img_path = r'timg.jpg'
    img = cv2.imread(img_path)
    h, w, c = img.shape
    split = np.zeros((h, 3, 3), dtype=np.uint8)
    split[:, :, :] = 255
    # rule_crop1 = rule_crop(img, box_ratio=(3. / 4, 3. / 4), location_type='LT', resize_w=w, resize_h=h)
    # rule_crop2 = rule_crop(img, box_ratio=(3. / 4, 3. / 4), location_type='RT', resize_w=w, resize_h=h)
    # rule_crop3 = rule_crop(img, box_ratio=(3. / 4, 3. / 4), location_type='LB', resize_w=w, resize_h=h)
    # rule_crop4 = rule_crop(img, box_ratio=(3. / 4, 3. / 4), location_type='RB', resize_w=w, resize_h=h)
    # rule_crop5 = rule_crop(img, box_ratio=(3. / 4, 3. / 4), location_type='CC', resize_w=w, resize_h=h)

    # img_rotate1 = random_rotate(img,(-10,10))
    # img_rotate1 = cv2.resize(img_rotate1,dsize=(w,h))
    # print(img_rotate1.shape)

    img_blur1 = gaussian_filter(img, (5, 5), 1.)
    img_blur2 = gaussian_filter(img, (5, 5), 20.)

    # img_all = np.hstack((img, split, img1, split, img2, split, img3))
    # img_all = np.hstack((img, split, img3, split, img4))
    # img_all = np.hstack((img, split, img5, split, img4))
    img_all = np.hstack((img, split, img_blur1, split, img_blur2))

    cv2.imshow('d', img_all)
    cv2.waitKey(0)


def Test2():
    import glob
    root_dir = r'D:/AI/DataSet/emotion/fer2013'
    train = root_dir + '/test_class_1'
    for f in glob.glob(train + '/*.jpg'):
        img = cv2.imread(f)
        img = cv2.resize(img, dsize=None, fx=3, fy=3)
        h, w, c = img.shape
        split = np.zeros((h, 3, 3), dtype=np.uint8)
        split[:, :, :] = 255

        # img_blur1 = gaussianBlue(img, (5, 5), 3.)
        img_br1 = adjust_contrast_bright(img, .8, 80)
        img_all = np.hstack((img, split, img_br1))
        cv2.imshow('d', img_all)
        cv2.waitKey(0)


import os.path as osp


def Test3():
    _path = '../dog.66.jpg'
    assert osp.exists(_path)
    img = cv2.imread(_path, 0)
    cv2.imshow('1', img)
    noise_groups = ('gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle')
    for i, t in enumerate(noise_groups):
        img_noise = noise_optional(img.copy(), t)
        cv2.imshow(f'noise_{t}', img_noise)
    # img_1 = salt_noise(img.copy(), 0.98)
    # cv2.imshow('2', img_1)
    # img_2 = gaussian_noise(img.copy(), 2, 8)
    # cv2.imshow('3', img_2)
    # img_3 = shift(img.copy(),10,10)
    # cv2.imshow('4', img_3)
    cv2.waitKey(0)


def Test4():
    _path = '/media/dangxs/E/Project/DataSet/Cat_vs_Dog/dog/dog.0.jpg'
    assert osp.exists(_path)
    img = cv2.imread(_path, 0)
    img1 = erase(img, 0.001, 0.01, values=[random.randint(0, 255)] * 3)
    cv2.imshow('1', img)
    cv2.imshow('2', img1)

    # img_hist = cv2.calcHist(img, [0], None, [256], [0, 255])
    # img1_hist = cv2.calcHist(img1, [0], None, [256], [0, 255])
    # plt.subplot(121)
    # plt.plot(img_hist)
    # plt.subplot(122)
    # plt.plot(img1_hist)
    # plt.show()

    cv2.waitKey(0)


if __name__ == '__main__':
    # gen_heatmap(50*2, 50)
    img = np.zeros([300, 400], np.uint8)
    gen_heatmap2(img, [[5, 30], [120, 250]], 10)
