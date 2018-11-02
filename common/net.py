#!usr/bin/python
# -*- coding: UTF-8 -*-

import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class Generator(Chain):

    def __init__(self, n_hidden, bottom_width=3, wscale=0.02):
        w = chainer.initializers.Normal(scale=wscale)
        self.n_hidden = n_hidden
        self.bottom_width = bottom_width
        super(Generator, self).__init__()
        with self.init_scope():
            self.z2l = L.Linear(None, 4 * 4 * 512, initialW=w)
            self.bn1 = L.BatchNormalization(512)
            self.deconv1 = L.Deconvolution2D(
                512, 256, ksize=4, stride=2, pad=1, outsize=(8, 8), initialW=w)
            self.bn2 = L.BatchNormalization(256)
            self.deconv2 = L.Deconvolution2D(
                256, 128, ksize=4, stride=2, pad=2, outsize=(14, 14), initialW=w)
            self.bn3 = L.BatchNormalization(128)
            self.deconv3 = L.Deconvolution2D(
                128, 1, ksize=4, stride=2, pad=1, outsize=(28, 28), initialW=w)

    def make_hidden(self, batchsize):
        return self.xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype("f")

    def __call__(self, x):
        h = F.reshape(F.relu(self.z2l(x)), (x.shape[0], 512, 4, 4))
        h = F.relu(self.deconv1(self.bn1(h)))
        h = F.relu(self.deconv2(self.bn2(h)))
        h = self.deconv3(self.bn3(h))
        return F.tanh(h)


class Discriminator(Chain):

    def __init__(self, output_dim=1):
        # self.in_channel = in_channel
        w = chainer.initializers.Normal(scale=0.02)
        super(Discriminator, self).__init__()
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
            self.lout = L.Linear(4 * 4 * 512, output_dim, initialW=w)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)), slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.2)
        h = F.reshape(h, (x.shape[0], -1))

        return self.lout(h)


class CifarGenerator(Chain):

    def __init__(self, n_hidden, wscale=0.02, ch=512, bottom_width=4):
        super(CifarGenerator, self).__init__()
        self.n_hidden = n_hidden
        self.output_activation = F.tanh
        self.hidden_activation = F.relu
        self.ch = ch
        self.bottom_width = bottom_width
        with self.init_scope():
            w = chainer.initializers.Normal(scale=wscale)
            self.l0 = L.Linear(None, bottom_width * bottom_width * ch,
                               initialW=w)
            self.dc1 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dc4 = L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w)
            # if self.use_bn:
            self.bn0 = L.BatchNormalization(bottom_width * bottom_width * ch)
            self.bn1 = L.BatchNormalization(ch // 2)
            self.bn2 = L.BatchNormalization(ch // 4)
            self.bn3 = L.BatchNormalization(ch // 8)

    def make_hidden(self, batchsize):
        return self.xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(self.xp.float32)
        # return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
        # .astype(np.float32)

    def __call__(self, z):
        h = F.reshape(self.hidden_activation(self.bn0(self.l0(z))),
                      (len(z), self.ch, self.bottom_width, self.bottom_width))
        h = self.hidden_activation(self.bn1(self.dc1(h)))
        h = self.hidden_activation(self.bn2(self.dc2(h)))
        h = self.hidden_activation(self.bn3(self.dc3(h)))
        x = self.output_activation(self.dc4(h))
        return x


class CifarDiscriminator(Chain):

    def __init__(self, wscale=0.02):
        super(CifarDiscriminator, self).__init__()
        w = chainer.initializers.Normal(scale=wscale)
        with self.init_scope():
            # input=(32,32), output=(32, 32)
            self.conv1 = L.Convolution2D(
                3, 32, ksize=3, stride=1, pad=0, initialW=w)
            self.conv2 = L.Convolution2D(
                32, 64, ksize=2, stride=2, pad=0, initialW=w)
            self.conv3 = L.Convolution2D(
                64, 128, ksize=2, stride=2, pad=0, initialW=w)
            self.conv4 = L.Convolution2D(
                128, 256, ksize=2, stride=2, pad=0, initialW=w)
            self.conv5 = L.Convolution2D(
                256, 512, ksize=3, stride=1, pad=1, initialW=w)

            self.bn1 = L.BatchNormalization(32)
            self.bn2 = L.BatchNormalization(64)
            self.bn3 = L.BatchNormalization(128)
            self.bn4 = L.BatchNormalization(256)
            self.bn5 = L.BatchNormalization(512)

            self.lout = L.Linear(None, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)), slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.2)
        h = F.leaky_relu(self.bn4(self.conv4(h)), slope=0.2)
        h = self.bn5(self.conv5(h))
        h = F.reshape(h, (x.shape[0], -1))

        return self.lout(h)


class WGANDiscriminator(chainer.Chain):
    def __init__(self, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        w = chainer.initializers.Normal(wscale)
        super(WGANDiscriminator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(None, ch // 8, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.c1_0 = L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, initialW=w)
            self.c2 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.c2_0 = L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, initialW=w)
            self.c3 = L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, initialW=w)
            self.c3_0 = L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, initialW=w)
            self.l4 = L.Linear(bottom_width * bottom_width *
                               ch, output_dim, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.c1(h))
        h = F.leaky_relu(self.c1_0(h))
        h = F.leaky_relu(self.c2(h))
        h = F.leaky_relu(self.c2_0(h))
        h = F.leaky_relu(self.c3(h))
        h = F.leaky_relu(self.c3_0(h))
        return self.l4(h)


class FCGenerator(Chain):

    def __init__(self, n_hidden):
        self.n_hidden = n_hidden
        super(FCGenerator, self).__init__()
        with self.init_scope():
            self.z2l = L.Linear(None, 128)
            self.l1 = L.Linear(None, 256)
            self.l2 = L.Linear(None, 2)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)

    def make_hidden(self, batchsize):
        return self.xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(self.xp.float32)

    def __call__(self, x):
        h = F.relu(self.z2l(x))
        h = F.relu(self.l1(self.bn1(h)))
        h = self.l2(self.bn2(h))
        return h


class FCDiscriminator(Chain):

    def __init__(self):
        super(FCDiscriminator, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 256)
            self.l2 = L.Linear(None, 128)
            self.l3 = L.Linear(None, 1)

            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(128)

    def __call__(self, x):
        h = F.leaky_relu(self.l1(x), slope=0.2)
        h = F.leaky_relu(self.l2(self.bn1(h)), slope=0.2)
        h = self.l3(self.bn2(h))

        return h
