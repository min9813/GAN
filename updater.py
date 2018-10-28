#!usr/bin/python
# -*- coding: UTF-8 -*-import chainer

import chainer
import chainer.functions as F
import warnings
from chainer import Variable
# import sys
warnings.filterwarnings('ignore')


class DCGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        print("successfully use DCGAN's updater")
        self.gen, self.dis = kwargs.pop('models')
        super(DCGANUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, y_fake, y_real, xp):
        batch_size = len(y_fake)
        L1 = F.sigmoid_cross_entropy(y_real, Variable(
            xp.ones(batch_size, dtype=xp.int32)).reshape(-1, 1))
        L2 = F.sigmoid_cross_entropy(y_fake, Variable(
            xp.zeros(batch_size, dtype=xp.int32)).reshape(-1, 1))
        loss = 0.5 * (L1 + L2)
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake, xp):
        batch_size = len(y_fake)
        loss = F.sigmoid_cross_entropy(y_fake, Variable(
            xp.ones(batch_size, dtype=xp.int32)).reshape(-1, 1))
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        x_real = Variable(self.converter(batch, self.device))
        xp = chainer.cuda.get_array_module(x_real.data)
        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        y_real = dis(x_real)
        z = gen.make_hidden(batchsize)
        x_fake = gen(z)
        y_fake = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real, xp)
        gen_optimizer.update(self.loss_gen, gen, y_fake, xp)


class CGANDCGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        print("successfully use CGAN's updater")
        self.gen, self.dis = kwargs.pop('models')
        self.class_num = kwargs.pop("class_num")
        self.xp = self.gen.xp
        super(CGANDCGANUpdater, self).__init__(*args, **kwargs)

    def make_label_infomation(self, label, x_data, batchsize):
        label = self.xp.array(label)
        real_label_image = self.xp.eye(self.class_num, dtype=self.xp.float32)[
            label][:, :, None, None]
        tmp_label_image = self.xp.ones(
            (batchsize, self.class_num, x_data.shape[-2], x_data.shape[-1]), dtype=self.xp.float32)
        real_label_image = tmp_label_image * real_label_image

        one_hot_label = self.xp.eye(self.class_num)[label].astype("f")
        one_hot_label = one_hot_label[:, :, None, None]
        x_data = self.xp.concatenate([x_data, real_label_image], axis=1)

        return one_hot_label, x_data, real_label_image

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        # print(batch)
        # sys.exit()
        batchsize = len(batch)
        x_real, real_label = zip(*batch)

        x_real = self.xp.asarray(x_real).astype("f")
        x_real = x_real * 2 - 1

        one_hot_label, x_real, real_label_image = self.make_label_infomation(
            real_label, x_real, batchsize)

        # make real_label to image

        # make fake one-hot label and fake label image
        # fake_label = self.xp.random.randint(0, self.class_num, size=batchsize)
        # fake_label_image = self.xp.zeros((
        # batchsize, self.class_num, x_real.shape[-2], x_real.shape[-1])).astype("f")
        # fake_label_image[:, fake_label, :, :] = 1.0
        # to concatenate with 4-dimension tensor 'z', add new two axis

        # combine image and real_label information

        x_real = Variable(x_real)

        y_real = self.dis(x_real)

        z = self.gen.make_hidden(batchsize)
        z = Variable(self.xp.concatenate([z, one_hot_label], axis=1))

        # all the axis except for concatenation axis must have same dimension.
        x_fake = self.gen(z)

        # combine fake image and real_label information
        x_fake = F.concat([x_fake, Variable(real_label_image)])
        y_fake = self.dis(x_fake)

        loss_gen = F.sigmoid_cross_entropy(
            y_fake, self.xp.ones_like(y_fake.data).astype("i"))

        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()

        x_fake.unchain_backward()

        loss_dis = F.sigmoid_cross_entropy(
            y_real, self.xp.ones_like(y_real.data).astype("i"))
        loss_dis += F.sigmoid_cross_entropy(y_fake,
                                            self.xp.zeros_like(y_fake.data).astype("i"))

        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'gen/loss': loss_gen})
        chainer.reporter.report({'dis/loss': loss_dis})


class WGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.generator, self.critic = kwargs.pop("models")
        self.opt_diter, self.change_step = kwargs.pop("step")
        super(WGANUpdater, self).__init__(*args, **kwargs)
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


class WGANGPUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop("models")
        self.n_dis, self.change_step = kwargs.pop("step")
        self.lam = kwargs.pop("gradient_penalty_weight")
        super(WGANGPUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('critic')
        xp = self.gen.xp

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batchsize = len(batch)
            x_real = Variable(self.converter(batch, self.device))
            y_real = self.dis(x_real)

            z = self.gen.make_hidden(batchsize)

            x_fake = self.gen(z)
            y_fake = self.dis(x_fake)

            if i == 0:
                loss_gen = F.mean(-y_fake)
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'gen/loss': loss_gen})
            x_fake.unchain_backward()

            eps = xp.random.uniform(0, 1, size=batchsize).astype("f")[
                :, None, None, None]
            x_mid = eps * x_real + (1.0 - eps) * x_fake

            grad, = chainer.grad([self.dis(x_mid)], [x_mid],
                                 enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))
            loss_gp = self.lam * \
                F.mean_squared_error(grad, xp.ones_like(grad.data))

            loss_dis = F.mean(-y_real)
            loss_dis += F.mean(y_fake)

            self.dis.cleargrads()
            loss_dis.backward()
            loss_gp.backward()
            dis_optimizer.update()

            chainer.reporter.report({'critic/loss': loss_dis + loss_gp})
            chainer.reporter.report({"wasserstein distance": -loss_dis})
            chainer.reporter.report({'critic/loss_grad': loss_gp})
            chainer.reporter.report({'g': F.mean(grad)})


class CGANWGANGPUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop("models")
        self.n_dis, self.change_step = kwargs.pop("step")
        self.lam = kwargs.pop("gradient_penalty_weight")
        self.class_num = kwargs.pop("class_num")
        self.xp = self.gen.xp
        super(CGANWGANGPUpdater, self).__init__(*args, **kwargs)

    def make_label_infomation(self, label, x_data, batchsize):
        label = self.xp.array(label)
        real_label_image = self.xp.eye(self.class_num, dtype=self.xp.float32)[
            label][:, :, None, None]
        tmp_label_image = self.xp.ones(
            (batchsize, self.class_num, x_data.shape[-2], x_data.shape[-1]), dtype=self.xp.float32)
        real_label_image = tmp_label_image * real_label_image

        one_hot_label = self.xp.eye(self.class_num)[label].astype("f")
        one_hot_label = one_hot_label[:, :, None, None]
        x_data = self.xp.concatenate([x_data, real_label_image], axis=1)

        return one_hot_label, x_data, real_label_image

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('critic')

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batchsize = len(batch)

            x_real, real_label = zip(*batch)
            x_real = self.xp.asarray(x_real).astype("f")
            x_real = x_real * 2 - 1

            one_hot_label, x_real, real_label_image = self.make_label_infomation(
                real_label, x_real, batchsize)

            x_real = Variable(x_real)

            y_real = self.dis(x_real)

            z = self.gen.make_hidden(batchsize)
            z = Variable(self.xp.concatenate([z, one_hot_label], axis=1))

            # all the axis except for concatenation axis must have same dimension.
            x_fake = self.gen(z)

            # combine fake image and real_label information
            x_fake = F.concat([x_fake, Variable(real_label_image)])
            y_fake = self.dis(x_fake)
            y_real = self.dis(x_real)

            if i == 0:
                loss_gen = F.mean(-y_fake)
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'gen/loss': loss_gen})
            x_fake.unchain_backward()

            eps = self.xp.random.uniform(0, 1, size=batchsize).astype("f")[
                :, None, None, None]
            x_mid = eps * x_real + (1.0 - eps) * x_fake

            grad, = chainer.grad([self.dis(x_mid)], [x_mid],
                                 enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))
            loss_gp = self.lam * \
                F.mean_squared_error(grad, self.xp.ones_like(grad.data))

            loss_dis = F.mean(-y_real)
            loss_dis += F.mean(y_fake)

            self.dis.cleargrads()
            loss_dis.backward()
            loss_gp.backward()
            dis_optimizer.update()

            chainer.reporter.report({'critic/loss': loss_dis + loss_gp})
            chainer.reporter.report({"wasserstein distance": -loss_dis})
            chainer.reporter.report({'critic/loss_grad': loss_gp})
            chainer.reporter.report({'g': F.mean(grad)})
