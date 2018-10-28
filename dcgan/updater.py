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
        super(Updater, self).__init__(*args, **kwargs)

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


class CGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        models = kwargs.pop("models")
        self.gen = models["gen"]
        self.dis = models["dis"]
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
        dis_optimizer = self.get_optimizer('dis')

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
