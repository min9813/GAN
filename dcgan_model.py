import chainer
from chainer import Chain
import chainer.functions as F
import chainer.links as L


class Generator(Chain):

    def __init__(self):
        w = chainer.initializers.Normal(scale=0.02)
        super(Generator, self).__init__()
        with self.init_scope():
            self.z2l = L.Linear(None, 4 * 4 * 512, initialW=w)
            self.bn1 = L.BatchNormalization(512)
            self.deconv1 = L.Deconvolution2D(
                512, 256, ksize=3, stride=2, outsize=(8, 8), initialW=w)
            self.bn2 = L.BatchNormalization(256)
            self.deconv2 = L.Deconvolution2D(
                256, 128, ksize=3, stride=2, outsize=(16, 16), initialW=w)
            self.bn3 = L.BatchNormalization(128)
            self.deconv3 = L.Deconvolution2D(
                128, 1, ksize=3, stride=2, pad=2, outsize=(28, 28), initialW=w)

    def __call__(self, x):
        h = F.reshape(F.relu(self.z2l(x)), (-1, 1024, 4, 4))
        h = F.relu(self.deconv1(F.bn1(h)))
        h = F.relu(self.deconv2(F.bn2(h)))
        h = F.relu(self.deconv3(F.bn3(h)))
        return F.tanh(h)


class Discriminator(Chain):

    def __init__(self):
        w = chainer.initializers.Normal(scale=0.02)
        super(Discriminator, self).__init__()
        with self.init_scope():
            # 28
            self.conv1 = L.Convolution2D(
                1, 128, ksize=4, stride=2, pad=1, initialW=w)
            # 14
            self.conv2 = L.Convolution2D(
                128, 256, ksize=4, stride=2, pad=1, initialW=w)
            # 7
            self.conv3 = L.Convolution2D(
                256, 512, ksize=3, stride=2, pad=1, initialW=w)
            # 3
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)))
        h = F.leaky_relu(self.bn2(self.conv2(x)))
        h = F.leaky_relu(self.bn3(self.conv3(x)))
        h = F.reshape(h, (x.shape[0], -1))

        return h


class CifarGenerator(Chain):

    def __init__(self, n_hidden, wscale=0.02):
        w = chainer.initializers.Normal(scale=wscale)
        self.n_hidden = n_hidden
        super(CifarGenerator, self).__init__()
        with self.init_scope():
            self.z2l = L.Linear(None, 4 * 4 * 512, initialW=w)
            self.deconv1 = L.Deconvolution2D(
                512, 256, ksize=2, stride=2, pad=0, outsize=(8, 8), initialW=w)
            self.deconv2 = L.Deconvolution2D(
                256, 128, ksize=2, stride=2, pad=0, outsize=(16, 16), initialW=w)
            self.deconv3 = L.Deconvolution2D(
                128, 64, ksize=2, stride=2, pad=0, outsize=(32, 32), initialW=w)
            self.deconv4 = L.Deconvolution2D(
                64, 3, ksize=3, stride=1, pad=1, outsize=(32, 32), initialW=w)

            self.bn1 = L.BatchNormalization(512)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(128)
            self.bn4 = L.BatchNormalization(64)

    def make_hidden(self, batchsize):
        return self.xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(self.xp.float32)

    def __call__(self, x):
        h = F.reshape(F.relu(self.z2l(x)), (-1, 512, 4, 4))
        h = F.relu(self.deconv1(self.bn1(h)))
        h = F.relu(self.deconv2(self.bn2(h)))
        h = F.relu(self.deconv3(self.bn3(h)))
        h = F.relu(self.deconv4(self.bn4(h)))
        return F.tanh(h)


class CifarDiscriminator(Chain):

    def __init__(self, wscale=0.02):
        super(CifarDiscriminator, self).__init__()
        w = chainer.initializers.Normal(scale=wscale)
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, ksize=2, stride=2, pad=0, initialW=w)
            self.conv2 = L.Convolution2D(
                64, 128, ksize=2, stride=2, pad=0, initialW=w)
            self.conv3 = L.Convolution2D(
                128, 256, ksize=2, stride=2, pad=0, initialW=w)
            self.conv4 = L.Convolution2D(
                256, 512, ksize=3, stride=1, pad=1, initialW=w)

            self.bn1 = L.BatchNormalization(64)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(256)
            self.bn4 = L.BatchNormalization(512)

            self.lout = L.Linear(None, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)), slope=0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.2)
        h = F.leaky_relu(self.bn4(self.conv4(h)), slope=0.2)
        h = F.reshape(h, (x.shape[0], -1))

        return self.lout(h)
