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
        self.n_dis, self.change_step = kwargs.pop("n_dis")
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        xp = self.gen.xp

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batchsize = len(batch)

            if i == 0:
                z = self.gen.make_hidden(batchsize)
                x_fake = self.gen(z)
                y_fake = self.dis(x_fake)
                loss_gen = F.sigmoid_cross_entropy(y_fake, Variable(
                    xp.ones_like(y_fake.data, dtype=xp.int8)))
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'gen/loss': loss_gen})
            x_fake.unchain_backward()
            x_real = Variable(self.converter(batch, self.device))
            y_real = self.dis(x_real)

            z = self.gen.make_hidden(batchsize)

            x_fake = self.gen(z)
            y_fake = self.dis(x_fake.data)
            loss_dis = F.sigmoid_cross_entropy(y_real, Variable(
                xp.ones_like(y_real.data, dtype=xp.int8)))
            loss_dis += F.sigmoid_cross_entropy(y_fake, Variable(
                xp.zeros_like(y_fake.data, dtype=xp.int8)))

            self.dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()

            chainer.reporter.report({'dis/loss': loss_dis})
