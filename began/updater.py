import chainer
from chainer import Variable


class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        models = kwargs.pop("models")
        self.gen = models["gen"]
        self.dis = models["dis"]
        self.opt_diter, self.change_step = kwargs.pop("n_dis")
        self.gamma = kwargs.pop("gamma")
        self.lambda_k = kwargs.pop("lambda_k")
        super(Updater, self).__init__(*args, **kwargs)
        self.xp = self.gen.xp
        self.kt = 0

    def update_core(self):
        gen_optimizer = self.get_optimizer("gen")
        dis_optimizer = self.get_optimizer("dis")
        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device))

        # xp = chainer.cuda.get_array_module(x_real.data)
        batchsize = len(batch)

        # maximize output of real data
        y_real = self.dis(x_real)

        z = self.gen.make_hidden(batchsize)
        x_fake = self.gen(z)
        # minimize output of fake data
        y_fake = self.dis(x_fake)

        loss_dis = y_real - self.kt * y_fake
        loss_gen = y_fake
        equilibrium_measure = self.gamma * y_real.data - y_fake.data
        self.kt = self.kt + self.lambda_k * equilibrium_measure
        # print(self.kt)
        self.kt = self.xp.clip(self.kt, 0, 1)

        measurement = y_real.array + self.xp.abs(equilibrium_measure)

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()

        x_fake.unchain_backward()

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({
            'dis/loss': loss_dis,
            'gen/loss': loss_gen,
            'kt': self.kt,
            'measurement': measurement})
