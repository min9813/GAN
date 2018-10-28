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
        self.n_dis, self.change_step = kwargs.pop("step")
        self.lam = kwargs.pop("gradient_penalty_weight")
        super(Updater, self).__init__(*args, **kwargs)

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


class CGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        models = kwargs.pop("models")
        self.gen = models["gen"]
        self.dis = models["dis"]
        self.n_dis, self.change_step = kwargs.pop("step")
        self.lam = kwargs.pop("gradient_penalty_weight")
        self.class_num = kwargs.pop("class_num")
        self.xp = self.gen.xp
        super(CGANUpdater, self).__init__(*args, **kwargs)

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
