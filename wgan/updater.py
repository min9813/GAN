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
        self.generator = models["gen"]
        self.critic = models["dis"]
        self.opt_diter, self.change_step = kwargs.pop("step")
        super(Updater, self).__init__(*args, **kwargs)
        self.generator_iterations = 0

    def update_core(self):
        generator_optimizer = self.get_optimizer("gen")
        critic_optimizer = self.get_optimizer("critic")
        critic_iter = self.opt_diter
        for i in range(critic_iter):
            batch = self.get_iterator('main').next()
            x_real = Variable(self.converter(batch, self.device))

            # xp = chainer.cuda.get_array_module(x_real.data)
            batchsize = len(batch)

            # maximize output of real data
            error_real = F.mean(self.critic(x_real))

            z = self.generator.make_hidden(batchsize)
            x_fake = self.generator(z)
            # minimize output of fake data
            error_fake = F.mean(self.critic(x_fake))
            wasserstein_distance = error_real - error_fake
            loss_critic = -wasserstein_distance

            self.critic.cleargrads()
            loss_critic.backward()
            critic_optimizer.update()
            chainer.reporter.report({
                'critic/loss': -loss_critic,
                'wasserstein distance': wasserstein_distance})

        # if (self.iteration > 0) and (self.iteration % critic_iter == 0):
        # z = self.generator.make_hidden(batchsize)
        # x_fake = self.generator(z)

        # maximize output of fake data
        loss_generator = -error_fake
        self.generator.cleargrads()
        loss_generator.backward()
        generator_optimizer.update()
        chainer.reporter.report({'gen/loss': loss_generator})
