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
        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()
        chainer.reporter.report({'gen/loss': loss_gen})

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


class CGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        models = kwargs.pop("models")
        self.gen = models["gen"]
        self.dis = models["dis"]
        self.class_num = kwargs.pop("class_num")
        self.n_dis = kwargs.pop("n_dis")
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
