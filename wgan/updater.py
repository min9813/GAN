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
        self.opt_diter, self.change_step = kwargs.pop("n_dis")
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer("gen")
        dis_optimizer = self.get_optimizer("dis")
        for i in range(self.opt_iter):
            batch = self.get_iterator('main').next()
            x_real = Variable(self.converter(batch, self.device))

            # xp = chainer.cuda.get_array_module(x_real.data)
            batchsize = len(batch)

            # maximize output of real data
            y_real = F.mean(self.dis(x_real))

            z = self.gen.make_hidden(batchsize)
            x_fake = self.gen(z)
            # minimize output of fake data
            y_fake = self.dis(x_fake)
            if i == 0:
                loss_gen = F.mean(-y_fake)
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'gen/loss': loss_gen})
            x_fake.unchain_backward()

            wasserstein_distance = F.mean(y_real - y_fake)
            loss_dis = -wasserstein_distance

            self.dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()
            chainer.reporter.report({
                'dis/loss': -loss_dis,
                'wasserstein distance': wasserstein_distance})

        # if (self.iteration > 0) and (self.iteration % dis_iter == 0):
        # z = self.generator.make_hidden(batchsize)
        # x_fake = self.generator(z)

        # maximize output of fake data
