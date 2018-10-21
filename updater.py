#!usr/bin/python
# -*- coding: UTF-8 -*-import chainer

import chainer
import chainer.functions as F
import warnings
warnings.filterwarnings('ignore')


class DCGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        print("successfully use DCGAN's updater")
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real, xp):
        batch_size = len(y_fake)
        L1 = F.sigmoid_cross_entropy(y_real, chainer.Variable(
            xp.ones(batch_size, dtype=xp.int32)).reshape(-1, 1))
        L2 = F.sigmoid_cross_entropy(y_fake, chainer.Variable(
            xp.zeros(batch_size, dtype=xp.int32)).reshape(-1, 1))
        loss = 0.5 * (L1 + L2)
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake, xp):
        batch_size = len(y_fake)
        loss = F.sigmoid_cross_entropy(y_fake, chainer.Variable(
            xp.ones(batch_size, dtype=xp.int32)).reshape(-1, 1))
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real = chainer.Variable(self.converter(batch, self.device))
        xp = chainer.cuda.get_array_module(x_real.data)
        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        y_real = dis(x_real)
        z = gen.make_hidden(batchsize)
        x_fake = gen(z)
        y_fake = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real, xp)
        gen_optimizer.update(self.loss_gen, gen, y_fake, xp)


class WGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.generator, self.critic = kwargs.pop("models")
        self.opt_diter, self.change_step = kwargs.pop("step")
        super(WGANUpdater, self).__init__(*args, **kwargs)
        self.generator_iterations = 0

    def update_core(self):
        generator_optimizer = self.get_optimizer("gen")
        critic_optimizer = self.get_optimizer("critic")

        if self.generator_iterations < 25 or self.generator_iterations % 500 == 0:
            critic_iter = self.change_step
        else:
            critic_iter = self.opt_diter
        batch = self.get_iterator('main').next()
        x_real = chainer.Variable(self.converter(batch, self.device))

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

        if (self.iteration > 0) and (self.iteration % critic_iter == 0):
            z = self.generator.make_hidden(batchsize)
            x_fake = self.generator(z)

            # maximize output of fake data
            loss_generator = -F.mean(self.critic(x_fake))
            self.generator.cleargrads()
            loss_generator.backward()
            generator_optimizer.update()
            chainer.reporter.report({'gen/loss': loss_generator})

        self.generator_iterations += 1


class WGANGPUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.generator, self.critic = kwargs.pop("models")
        self.opt_diter, self.change_step = kwargs.pop("step")
        self.gradient_penalty_weight = kwargs.pop("gradient_penalty_weight")
        super(WGANGPUpdater, self).__init__(*args, **kwargs)
        self.generator_iterations = 0

    def update_core(self):
        generator_optimizer = self.get_optimizer("gen")
        critic_optimizer = self.get_optimizer("critic")

        if self.generator_iterations < 25 or self.generator_iterations % 500 == 0:
            critic_iter = self.change_step
        else:
            critic_iter = self.opt_diter
        batch = self.get_iterator('main').next()
        x_real = chainer.Variable(self.converter(batch, self.device))
        # xp = chainer.cuda.get_array_module(x_real.data)
        batchsize = len(batch)

        # maximize output of real data
        error_real = F.mean(self.critic(x_real))

        z = self.generator.make_hidden(batchsize)
        xp = chainer.cuda.get_array_module(z)

        x_fake = self.generator(z)

        e = xp.random.uniform(0, 1, (batchsize, 1, 1, 1))

        x_hat = e * x_real + (1 - e) * x_fake

        grad, = chainer.grad([self.critic(x_hat)], [x_hat],
                             enable_double_backprop=True)

        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        loss_grad = self.gradient_penalty_weight * \
            F.mean_squared_error(grad, xp.ones_like(grad.data))

        # minimize output of fake data
        error_fake = F.mean(self.critic(x_fake))
        wasserstein_distance = error_real - error_fake
        loss_critic = -wasserstein_distance + loss_grad

        self.critic.cleargrads()
        loss_critic.backward()
        critic_optimizer.update()
        chainer.reporter.report({
            'critic/loss_grad': loss_grad,
            'wasserstein distance': wasserstein_distance,
            'critic/loss': loss_critic})

        if (self.iteration > 0) and (self.iteration % critic_iter == 0):
            z = self.generator.make_hidden(batchsize)
            x_fake = self.generator(z)

            # maximize output of fake data
            loss_generator = -F.mean(self.critic(x_fake))
            self.generator.cleargrads()
            loss_generator.backward()
            generator_optimizer.update()
            chainer.reporter.report({'gen/loss': loss_generator})

        self.generator_iterations += 1
