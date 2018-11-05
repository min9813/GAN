#!usr/bin/python
# -*- coding: UTF-8 -*-import chainer

import chainer
import chainer.functions as F
import warnings
from chainer import Variable
# import sys
warnings.filterwarnings('ignore')


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        models = kwargs.pop("models")
        self.gen = models["gen"]
        self.dis = models["dis"]
        self.n_dis = kwargs.pop("n_dis")

        self.xp = self.gen.xp
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device))
        batchsize = len(batch)
        z = self.gen.make_hidden(batchsize)
        x_fake = self.gen(z)
        y_real = self.dis(x_real)
        y_fake = self.dis(x_fake)

        loss_gen = F.sigmoid_cross_entropy(y_fake, Variable(
            self.xp.ones_like(y_fake.data, dtype=self.xp.int32)))
        loss_feature = F.mean_squared_error(y_fake, y_real.data)
        self.gen.cleargrads()
        loss_gen.backward()
        loss_feature.backward()
        gen_optimizer.update()
        chainer.reporter.report({'gen/loss': loss_gen})
        chainer.reporter.report({'gen/loss_feature': loss_feature})

        x_fake.unchain_backward()
        L1 = F.sigmoid_cross_entropy(y_real, Variable(
            self.xp.ones_like(y_real.data, dtype=self.xp.int32)))
        L2 = F.sigmoid_cross_entropy(y_fake, Variable(
            self.xp.zeros_like(y_fake.data, dtype=self.xp.int32)))
        loss_dis = L1 + L2
        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'dis/loss': loss_dis})
