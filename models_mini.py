"""
使用mobilenet_v2 作为骨干网络
"""

from torch.nn import functional as F
from torch.nn import init
import math
from mobilenet_v2 import *


class HyperNet(nn.Module):
    """
    Hyper network for learning perceptual rules.

    Args:
        lda_out_channels: local distortion aware module output size.
        hyper_in_channels: input feature channels for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        feature_size: input feature map width/height for hyper network.

    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.

    """

    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size, target_fc1_size, target_fc2_size,
                 target_fc3_size, target_fc4_size, feature_size, backbone='mobilenet_v2', to_onnx=False):
        super(HyperNet, self).__init__()

        assert backbone =='mobilenet_v2'

        self.to_onnx_flag = to_onnx
        self.hyperInChn = hyper_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size

        _output_ch = 1280
        self.res = MobileNetV2Backbone(lda_out_channels, target_in_size, is_pretrain=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Conv layers for resnet output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(_output_ch, _output_ch // 2, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(_output_ch // 2, _output_ch // 4, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(_output_ch // 4, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,
                                   padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def f(self, img):
        feature_size = self.feature_size

        res_out = self.res(img)

        # input vector for target net
        target_in_vec = res_out['target_in_vec'].view(-1, self.target_in_size, 1, 1)

        # input features for hyper net
        hyper_in_feat = self.conv1(res_out['hyper_in_feat']).view(-1, self.hyperInChn, feature_size, feature_size)

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b
        return out

    def forward(self, img):
        if self.to_onnx_flag:
            return self.onnx_forward(img)

        return self.f(img)

    def onnx_fc(self, input_, weight, bias):
        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = weight.view(weight.shape[0] * weight.shape[1], weight.shape[2],
                                weight.shape[3], weight.shape[4])
        bias_re = bias.view(bias.shape[0] * bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=weight.shape[0])
        out = out.view(input_.shape[0], weight.shape[1], input_.shape[2], input_.shape[3])
        return out

    def onnx_forward(self, img):
        out = self.f(img)
        # return out
        ########################################################################################
        # replace TargetNet.forward()
        # x = out['target_in_vec']
        # for i in range(1, 6):
        #     a = f'target_fc{i}w'
        #     b = f'target_fc{i}b'
        #     x = self.onnx_fc(x, out[a], out[b])
        #     if i < 5: x = torch.sigmoid(x)
        # x = torch.squeeze(x)
        # return x

        outs = [
            out['target_in_vec'],
            out['target_fc1w'], out['target_fc1b'],
            out['target_fc2w'], out['target_fc2b'],
            out['target_fc3w'], out['target_fc3b'],
            out['target_fc4w'], out['target_fc4b'],
            out['target_fc5w'], out['target_fc5b'],
        ]
        return self.postprocess(outs)

    def postprocess(self, outs):
        x = outs[0]
        for i in range(1, 11, 2):
            dim0 = x.shape[0]
            dim2 = x.shape[2]
            dim3 = x.shape[3]
            x1 = torch.reshape(x, [-1, dim0 * x.shape[1], dim2, dim3])
            w, b = outs[i], outs[i + 1]
            w1 = torch.reshape(w, [w.shape[0] * w.shape[1], *w.shape[2:]])
            b1 = b.reshape(-1)
            x1 = torch.conv2d(x1, w1, b1, groups=w.shape[0])
            x = torch.reshape(x1, [dim0, w.shape[1], dim2, dim3])
            if i < 9:  x = torch.sigmoid(x)
        return x


class TargetNet(nn.Module):
    """
    Target network for quality prediction.
    """

    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.l1 = nn.Sequential(
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = nn.Sequential(
            TargetFC(paras['target_fc3w'], paras['target_fc3b']),
            nn.Sigmoid(),
        )

        self.l4 = nn.Sequential(
            TargetFC(paras['target_fc4w'], paras['target_fc4b']),
            nn.Sigmoid(),
            TargetFC(paras['target_fc5w'], paras['target_fc5b']),
        )

    def forward(self, x):
        q = self.l1(x)
        # q = F.dropout(q)
        q = self.l2(q)
        q = self.l3(q)
        q = self.l4(q).squeeze()
        return q


class TargetFC(nn.Module):
    """
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """

    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):
        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2],
                                     self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MobileNetV2Backbone(nn.Module):
    def __init__(self, lda_out_channels, in_chn, is_pretrain=True):
        super(MobileNetV2Backbone, self).__init__()
        self.backbone = MobileNetV2(widen_factor=1, out_indices=(1, 2, 4, 7))

        """
        0 torch.Size([1, 16, 112, 112])
        1 torch.Size([1, 24, 56, 56])   -->
        2 torch.Size([1, 32, 28, 28])   -->
        3 torch.Size([1, 64, 14, 14])
        4 torch.Size([1, 96, 14, 14])   -->
        5 torch.Size([1, 160, 7, 7])
        6 torch.Size([1, 320, 7, 7])
        7 torch.Size([1, 1280, 7, 7])   -->
        """

        # local distortion aware module
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(24, 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(4 * 64, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(16 * 16, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(1280, in_chn - lda_out_channels * 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # initialize
        nn.init.kaiming_normal_(self.lda1_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda2_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda3_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda1_fc.weight.data)
        nn.init.kaiming_normal_(self.lda2_fc.weight.data)
        nn.init.kaiming_normal_(self.lda3_fc.weight.data)
        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

        if is_pretrain:
            self.backbone.load_pretrained_weight()

        # # 冻结主干
        # for n, m in self.named_parameters():
        #     m.requires_grad = False

    def forward(self, x):

        outs = self.backbone(x)
        x1, x2, x3, x4 = outs

        # the same effect as lda operation in the paper, but save much more memory
        lda_1 = self.lda1_fc(self.lda1_pool(x1).view(x1.size(0), -1))
        # print('2-pool:', self.lda2_pool(x).shape)
        lda_2 = self.lda2_fc(self.lda2_pool(x2).view(x2.size(0), -1))
        # print('3-pool:', self.lda3_pool(x).shape)
        lda_3 = self.lda3_fc(self.lda3_pool(x3).view(x3.size(0), -1))
        # print('4-pool:', self.lda4_pool(x).shape)
        lda_4 = self.lda4_fc(self.lda4_pool(x4).view(x4.size(0), -1))

        vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)

        # print(x4.shape)      # [1,1280,7,7]
        # print(vec.shape)     # [1,224]

        out = {}
        out['hyper_in_feat'] = x4
        out['target_in_vec'] = vec

        return out


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    m = MobileNetV2Backbone(16, 224)
    # m = HyperNet(16, 112, 224, 112, 56, 28, 14, 7, backbone='mobilenet_v2')
    x = torch.randn(1, 3, 224, 224)



    _ = m(x)
