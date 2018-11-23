import chainer
import chainer.functions as F
import chainer.links as L
import common.sn.max_sv as max_sv


class SNConvolution2D(L.Convolution2D):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, use_gamma=False, Ip=1):
        self.Ip = Ip
        self.u = None
        self.use_gamma = use_gamma
        super(SNConvolution2D, self).__init__(
            in_channels, out_channels, ksize, stride, pad,
            nobias, initialW, initial_bias)

    @property
    def W_bar(self):
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_sv.max_singular_value(W_mat, self.u, self.Ip)
        sigma = F.broadcast_to(sigma.reshape((1, 1, 1, 1)), self.W.shape)
        self.u = _u
        if hasattr(self, 'gamma'):
            return F.broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNConvolution2D, self)._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = self.xp.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1, 1, 1))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return F.convolution_2d(
            x, self.W_bar, self.b, self.stride, self.pad)
