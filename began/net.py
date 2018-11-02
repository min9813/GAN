import chainer.functions as F
import chainer.links as L
import chainer


class CifarDiscriminator(chainer.Chain):

    def __init__(self, wscale=0.02, ch=512, metrics_dim=1):
        w = chainer.initializers.Normal(wscale)
        self.ch = ch
        if metrics_dim == 1:
            self.metric = F.mean_absolute_error
        else:
            self.metric = F.mean_squared_error
        super(CifarDiscriminator, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(None, ch // 8, 3, 1, 1, initialW=w)
            self.conv1 = L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, initialW=w)
            self.conv2 = L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, initialW=w)
            self.conv3 = L.Convolution2D(ch // 2, ch, 4, 2, 1, initialW=w)

            self.l0 = L.Linear(4*4*ch, ch // 4, initialW=w)
            self.l1 = L.Linear(ch // 4, 4*4*ch, initialW=w)

            self.dconv0 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dconv1 = L.Deconvolution2D(
                ch // 2, ch // 4, 4, 2, 1, initialW=w)
            self.dconv2 = L.Deconvolution2D(
                ch // 4, ch // 8, 4, 2, 1, initialW=w)
            self.dconv3 = L.Deconvolution2D(ch // 8, 3, 3, 1, 1, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.conv0(x))
        h = F.leaky_relu(self.conv1(h))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.l0(h))
        h = F.leaky_relu(self.l1(h))
        h = F.reshape(h, (x.data.shape[0], self.ch, 4, 4))
        h = F.leaky_relu(self.dconv0(h))
        h = F.leaky_relu(self.dconv1(h))
        h = F.leaky_relu(self.dconv2(h))
        h = self.dconv3(h)
        h = F.tanh(h)

        return self.metric(h, x)


class MnistDiscriminator(chainer.Chain):

    def __init__(self, wscale=0.02, ch=512, metrics_dim=1):
        w = chainer.initializers.Normal(wscale)
        self.ch = ch
        if metrics_dim == 1:
            self.metric = F.mean_absolute_error
        else:
            self.metric = F.mean_squared_error
        super(MnistDiscriminator, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(
                None, ch // 4, ksize=4, stride=2, pad=1, initialW=w)
            # 14
            self.conv1 = L.Convolution2D(
                ch // 4, ch // 2, ksize=4, stride=2, pad=2, initialW=w)
            # 8
            self.conv2 = L.Convolution2D(
                ch // 2, ch, ksize=4, stride=2, pad=1, initialW=w)

            self.l0 = L.Linear(ch*4*4, ch // 3, initialW=w)
            self.l1 = L.Linear(ch // 3, ch*4*4, initialW=w)

            self.dconv0 = L.Deconvolution2D(ch, ch // 2, 4, 2, 1, initialW=w)
            self.dconv1 = L.Deconvolution2D(
                ch // 2, ch // 4, 4, 2, 2, initialW=w)
            self.dconv2 = L.Deconvolution2D(ch // 4, 1, 4, 2, 1, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.conv0(x))
        h = F.leaky_relu(self.conv1(h))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.l0(h))
        h = F.leaky_relu(self.l1(h))
        h = F.reshape(h, (x.data.shape[0], self.ch, 4, 4))
        h = F.leaky_relu(self.dconv0(h))
        h = F.leaky_relu(self.dconv1(h))
        h = self.dconv2(h)
        h = F.tanh(h)

        return self.metric(h, x)


class CifarGenerator(chainer.Chain):

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

    def make_hidden(self, batchsize):
        return self.xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype(self.xp.float32)
        # return np.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)) \
        # .astype(np.float32)

    def __call__(self, z):
        h = F.reshape(self.hidden_activation(self.l0(z)),
                      (len(z), self.ch, self.bottom_width, self.bottom_width))
        h = self.hidden_activation(self.dc1(h))
        h = self.hidden_activation(self.dc2(h))
        h = self.hidden_activation(self.dc3(h))
        h = self.output_activation(self.dc4(h))
        return h


class MnistGenerator(chainer.Chain):

    def __init__(self, n_hidden, bottom_width=3, wscale=0.02):
        w = chainer.initializers.Normal(scale=wscale)
        self.n_hidden = n_hidden
        self.bottom_width = bottom_width
        super(MnistGenerator, self).__init__()
        with self.init_scope():
            self.z2l = L.Linear(None, 4 * 4 * 512, initialW=w)
            self.deconv1 = L.Deconvolution2D(
                512, 256, ksize=4, stride=2, pad=1, outsize=(8, 8), initialW=w)
            self.deconv2 = L.Deconvolution2D(
                256, 128, ksize=4, stride=2, pad=2, outsize=(14, 14), initialW=w)
            self.deconv3 = L.Deconvolution2D(
                128, 1, ksize=4, stride=2, pad=1, outsize=(28, 28), initialW=w)

    def make_hidden(self, batchsize):
        return self.xp.random.uniform(-1, 1, (batchsize, self.n_hidden, 1, 1)).astype("f")

    def __call__(self, x):
        h = F.reshape(F.relu(self.z2l(x)), (x.shape[0], 512, 4, 4))
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = self.deconv3(h)
        return F.tanh(h)
