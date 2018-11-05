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
        print("use loss function as sigmoid_cross_entropy")
        models = kwargs.pop("models")
        self.gen = models["gen"]
        self.dis = models["dis"]
        self.n_dis, self.change_step = kwargs.pop("n_dis")
        self.lam = kwargs.pop("gradient_penalty_weight")
        self.perturb_weight = kwargs.pop("perturb_weight")
        self.xp = self.gen.xp
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        xp = self.gen.xp

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batchsize = len(batch)
            x_real = self.converter(batch, self.device)
            x_real_std = self.xp.std(x_real, axis=0, keepdims=True)
            x_rnd = self.xp.random.uniform(0, 1, x_real.shape).astype("f")
            y_real = self.dis(x_real)

            z = self.gen.make_hidden(batchsize)
            x_fake = self.gen(z)
            y_fake = self.dis(x_fake)

            if i == 0:
                loss_gen = F.sigmoid_cross_entropy(
                    y_fake, xp.ones_like(y_fake.data).astype("int8"))
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'gen/loss': loss_gen})
            x_fake.unchain_backward()

            alpha = self.xp.random.uniform(0, 1, size=batchsize).astype("f")[
                :, None, None, None]
            x_perturb = Variable(x_real + alpha * self.perturb_weight * x_real_std * x_rnd)

            grad, = chainer.grad([self.dis(x_perturb)], [x_perturb],
                                 enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))
            loss_gp = self.lam * \
                F.mean_squared_error(grad, xp.ones_like(grad.data))

            loss_dis = F.sigmoid_cross_entropy(y_real, xp.ones_like(y_fake.data).astype("int8"))
            loss_dis += F.sigmoid_cross_entropy(y_fake, xp.zeros_like(y_fake.data).astype("int8"))

            self.dis.cleargrads()
            loss_dis.backward()
            loss_gp.backward()
            dis_optimizer.update()

            chainer.reporter.report({'dis/loss': loss_dis})
            chainer.reporter.report({'dis/loss_grad': loss_gp})
            chainer.reporter.report({'g': F.mean(grad)})


class CGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        models = kwargs.pop("models")
        self.gen = models["gen"]
        self.dis = models["dis"]
        self.n_dis, self.change_step = kwargs.pop("n_dis")
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

        # in order to avoid uneccessary backward history,
        # use xp.concat instead of F.concat and don't change each data to Variable
        x_data = self.xp.concatenate([x_data, real_label_image], axis=1)

        return one_hot_label, x_data, real_label_image

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        xp = self.gen.xp

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batchsize = len(batch)

            x_real, real_label = zip(*batch)
            x_real = self.xp.asarray(x_real).astype("f")

            x_real_std = self.xp.std(x_real, axis=0, keepdims=True)
            x_rnd = self.xp.random.uniform(0, 1, x_real.shape).astype("f")

            one_hot_label, x_real, real_label_image = self.make_label_infomation(
                real_label, x_real, batchsize)

            real_label_image = Variable(real_label_image)
            x_real = Variable(x_real)
            y_real = self.dis(x_real)

            z = self.gen.make_hidden(batchsize)
            z = Variable(self.xp.concatenate([z, one_hot_label], axis=1))
            x_fake = self.gen(z)
            x_fake = F.concat([x_fake, real_label_image])
            y_fake = self.dis(x_fake)

            if i == 0:
                loss_gen = F.sigmoid_cross_entropy(
                    y_fake, self.xp.ones_like(y_fake.data))
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'gen/loss': loss_gen})

            x_fake.unchain_backward()
            alpha = self.xp.random.uniform(0, 1, size=batchsize).astype("f")[
                :, None, None, None]
            x_perturb = x_real + alpha * self.perturb_range * x_real_std * x_rnd

            grad, = chainer.grad([self.dis(x_perturb)], [x_perturb],
                                 enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))
            loss_gp = self.lam * \
                F.mean_squared_error(grad, xp.ones_like(grad.data))

            loss_dis = F.sigmoid_cross_entropy(y_real, self.xp.ones_like(y_fake.data))
            loss_dis += F.sigmoid_cross_entropy(y_fake, self.zeros_like(y_fake.data))

            self.dis.cleargrads()
            loss_dis.backward()
            loss_gp.backward()
            dis_optimizer.update()

            chainer.reporter.report({'dis/loss': loss_dis})
            chainer.reporter.report({'dis/loss_grad': loss_gp})
            chainer.reporter.report({'g': F.mean(grad)})
