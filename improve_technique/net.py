#!usr/bin/python
# -*- coding: UTF-8 -*-

import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class MnistMinibatchDiscriminator(Chain):

    def __init__(self, output_dim=1, B=100, C=5, use_feature_matching=True):
        # self.in_channel = in_channel
        self.B = B
        self.C = C
        self.use_feature_matching = use_feature_matching
        w = chainer.initializers.Normal(scale=0.02)
        super(MnistMinibatchDiscriminator, self).__init__()
        with self.init_scope():
            # 28
            self.conv1 = L.Convolution2D(
                None, 128, ksize=4, stride=2, pad=1, initialW=w)
            # 14
            self.conv2 = L.Convolution2D(
                128, 256, ksize=4, stride=2, pad=2, initialW=w)
            # 8
            self.conv3 = L.Convolution2D(
                256, 512, ksize=4, stride=2, pad=1, initialW=w)
            # 4
            self.l_hidden = L.Linear(4 * 4 * 512, self.B * self.C, initialW=w)
            self.lout = L.Linear(4 * 4 * 512 + self.B, output_dim, initialW=w)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)

    def __call__(self, x):
        N = x.shape[0]
        h = F.leaky_relu(self.bn1(self.conv1(x)), slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.2)
        h = F.reshape(h, (x.shape[0], -1))
        feature = F.reshape(self.l_hidden(h), (N, self.B, self.C, 1))
        feature = F.broadcast_to(feature, (N, self.B, self.C, N))
        feature_batch = F.transpose(feature, (3, 1, 2, 0))
        feature = F.absolute(feature - feature_batch)
        feature = F.exp(-F.sum(feature, axis=2))
        feature = F.sum(feature, axis=2) - 1
        h = F.concat([h, feature])

        if self.use_feature_matching:
            return h, self.lout(h)
        else:
            return self.lout(h)


class CifarMinibatchDiscriminator(Chain):

    def __init__(self, wscale=0.02, ch=512, B=100, C=5, output_dim=1, use_feature_matching=True):
        super(CifarMinibatchDiscriminator, self).__init__()
        self.B = B
        self.C = C
        self.use_feature_matching = use_feature_matching
        w = chainer.initializers.Normal(scale=wscale)
        with self.init_scope():
            # input=(32,32), output=(32, 32)
            self.conv1 = L.Convolution2D(
                3, ch // 8, ksize=3, stride=1, pad=1, initialW=w)
            self.conv2 = L.Convolution2D(
                ch // 8, ch // 4, ksize=4, stride=2, pad=1, initialW=w)
            self.conv3 = L.Convolution2D(
                ch // 4, ch // 2, ksize=4, stride=2, pad=1, initialW=w)
            self.conv4 = L.Convolution2D(
                ch // 2, ch, ksize=4, stride=2, pad=1, initialW=w)

            self.bn1 = L.BatchNormalization(ch // 8)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 2)
            self.bn4 = L.BatchNormalization(ch)

            self.l_hidden = L.Linear(4 * 4 * ch, self.B * self.C, initialW=w)
            self.lout = L.Linear(4 * 4 * ch + self.B, output_dim, initialW=w)

    def __call__(self, x):
        N = x.shape[0]
        h = F.leaky_relu(self.bn1(self.conv1(x)), slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.2)
        h = F.leaky_relu(self.bn4(self.conv4(h)))
        h = F.reshape(h, (N, -1))
        feature = F.reshape(self.l_hidden(h), (N, self.B, self.C, 1))
        feature = F.broadcast_to(feature, (N, self.B, self.C, N))
        feature_batch = F.transpose(feature, (3, 1, 2, 0))
        feature = F.absolute(feature - feature_batch)
        feature = F.exp(-F.sum(feature, axis=2))
        feature = F.sum(feature, axis=2) - 1
        h = F.concat([h, feature])
        if self.use_feature_matching:
            return h, self.lout(h)
        else:
            return self.lout(h)


class CifarDeepMinibatchDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1, B=100, C=5):
        w = chainer.initializers.Normal(wscale)
        self.B = B
        self.C = C
        super(CifarDeepMinibatchDiscriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = L.Convolution2D(None, ch // 8, 3, 1, 1, initialW=w)
            self.c0_1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c1_1 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c2_1 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width *
                               ch, output_dim, initialW=w)

            self.l_hidden = L.Linear(
                bottom_width * bottom_width * ch, B * C, initialW=w)
            self.lout = L.Linear(
                bottom_width * bottom_width * ch + B, 1, initialW=w)
            self.bn0_1 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_0 = L.BatchNormalization(ch // 4, use_gamma=False)
            self.bn1_1 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_0 = L.BatchNormalization(ch // 2, use_gamma=False)
            self.bn2_1 = L.BatchNormalization(ch // 1, use_gamma=False)
            self.bn3_0 = L.BatchNormalization(ch // 1, use_gamma=False)

    def __call__(self, x):
        N = x.data.shape[0]
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.bn0_1(self.c0_1(h)))
        h = F.leaky_relu(self.bn1_0(self.c1_0(h)))
        h = F.leaky_relu(self.bn1_1(self.c1_1(h)))
        h = F.leaky_relu(self.bn2_0(self.c2_0(h)))
        h = F.leaky_relu(self.bn2_1(self.c2_1(h)))
        h = F.reshape(F.leaky_relu(self.c3_0(h)), (N, 8192))
        h = self.minibatch_discriminate(h, N)
        return self.lout(h)

    def minibatch_discriminate(self, h, N):
        feature = F.reshape(self.l_hidden(h), (N, self.B, self.C, 1))
        feature = F.broadcast_to(feature, (N, self.B, self.C, N))
        feature_batch = F.transpose(feature, (3, 1, 2, 0))
        feature = F.absolute(feature - feature_batch)
        feature = F.exp(-F.sum(feature, axis=2))
        feature = F.sum(feature, axis=2) - 1
        return F.concat([h, feature])
