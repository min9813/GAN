import chainer
import chainer.functions as F
import chainer.links as L
import common.sn.max_sv as max_sv


class SNLinear(L.Linear):

    def __init__(self, in_size, out_size, use_gamma=False, nobias=False,
                 initialW=None, initial_bias=None, Ip=1):
        self.Ip = Ip
        self.u = None
        self.use_gamma = use_gamma
        super(SNLinear, self).__init__(
            in_size, out_size, nobias, initialW, initial_bias
        )

    @property
    def W_bar(self):
        sigma, _u, _ = max_sv.max_singular_value(self.W, self.u, self.Ip)
        sigma = F.broadcast_to(sigma.reshape((1, 1)), self.W.shape)
        self.u = _u
        if hasattr(self, 'gamma'):
            return F.broadcast_to(self.gamma, self.W.shape) * self.W / sigma
        else:
            return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNLinear, self)._initialize_params(in_size)
        if self.use_gamma:
            _, s, _ = self.xp.linalg.svd(self.W.data)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        return F.linear(x, self.W_bar, self.b)
