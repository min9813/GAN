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
        self.lam = kwargs.pop("gradient_penalty_weight")
        self.xp = self.gen.xp
        super(Updater, self).__init__(*args, **kwargs)

    def l2_distance(self, tensor1, tensor2):
        return F.sqrt(F.sum((tensor1 - tensor2) ** 2, axis=1, keepdims=True))

    def critic(self, h_real, h_fake2):
        loss = self.l2_distance(h_real, h_fake2)
        zero_tensor = self.xp.zeros_like(h_real.data)
        loss -= self.l2_distance(h_real, zero_tensor)
        return loss

    def energy_distance(self, h_real, h_fake, h_fake2):
        loss = self.l2_distance(h_real, h_fake)
        loss += self.l2_distance(h_real, h_fake2)
        loss -= self.l2_distance(h_fake, h_fake2)
        return F.mean(loss)

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        xp = self.gen.xp

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batchsize = len(batch)
            x_real = Variable(self.converter(batch, self.device))
            h_real = self.dis(x_real)

            z = self.gen.make_hidden(batchsize)
            x_fake = self.gen(z)
            h_fake = self.dis(x_fake)

            z2 = self.gen.make_hidden(batchsize)
            x_fake2 = self.gen(z2)
            h_fake2 = self.dis(x_fake2)

            if i == 0:
                loss_gen = self.energy_distance(h_real, h_fake, h_fake2)
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'gen/loss': loss_gen})
            x_fake.unchain_backward()
            x_fake2.unchain_backward()

            critic_real = self.critic(h_real, h_fake2)
            critic_fake = self.critic(h_fake, h_fake2)

            loss_surrogate = F.mean(critic_real - critic_fake)

            eps = self.xp.random.uniform(0, 1, size=batchsize).astype("f")[
                :, None, None, None]
            x_mid = eps * x_real + (1.0 - eps) * x_fake

            h_mid = chainer.Variable(self.dis(x_mid).data)

            base_grad, = chainer.grad([self.critic(h_mid, h_fake.data)], [
                                      h_mid], enable_double_backprop=True)
            grad, = chainer.grad([self.dis(x_mid)], [x_mid], grad_outputs=[
                                 base_grad], enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))
            loss_gp = self.lam * \
                F.mean_squared_error(grad, xp.ones_like(grad.data))

            self.dis.cleargrads()
            (-loss_surrogate).backward()
            loss_gp.backward()
            dis_optimizer.update()

            chainer.reporter.report({'critic/loss': -loss_surrogate + loss_gp})
            chainer.reporter.report({"cramer distance": loss_surrogate})
            chainer.reporter.report({'critic/loss_grad': loss_gp})
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

    def l2_distance(self, tensor1, tensor2):
        return F.sqrt(F.batch_l2_norm_squared((tensor1 - tensor2)))

    def critic(self, h_real, h_fake2):
        loss = self.l2_distance(h_real, h_fake2)
        zero_tensor = self.xp.zeros_like(h_real.data)
        loss -= self.l2_distance(h_real, zero_tensor)
        return loss

    def energy_distance(self, h_real, h_fake, h_fake2):
        loss = self.l2_distance(h_real, h_fake)
        loss += self.l2_distance(h_real, h_fake2)
        loss -= self.l2_distance(h_fake, h_fake2)
        return F.mean(loss)

    def critic_process(self, h_mid, h_fake2):
        h_mid = self.dis(h_mid)
        h_mid = self.critic(h_mid, h_fake2)
        return h_mid

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')
        xp = self.gen.xp

        for i in range(self.n_dis):
            batch = self.get_iterator('main').next()
            batchsize = len(batch)

            x_real, real_label = zip(*batch)
            x_real = self.xp.asarray(x_real).astype("f")

            one_hot_label, x_real, real_label_image = self.make_label_infomation(
                real_label, x_real, batchsize)

            real_label_image = Variable(real_label_image)
            x_real = Variable(x_real)
            h_real = self.dis(x_real)

            z = self.gen.make_hidden(batchsize)
            z = Variable(self.xp.concatenate([z, one_hot_label], axis=1))
            x_fake = self.gen(z)
            x_fake = F.concat([x_fake, real_label_image])
            h_fake = self.dis(x_fake)

            z2 = self.gen.make_hidden(batchsize)
            z2 = Variable(self.xp.concatenate([z, one_hot_label], axis=1))
            x_fake2 = self.gen(z2)
            x_fake2 = F.concat([x_fake2, real_label_image])
            h_fake2 = self.dis(x_fake2)

            if i == 0:
                loss_gen = self.energy_distance(h_real, h_fake, h_fake2)
                self.gen.cleargrads()
                loss_gen.backward()
                gen_optimizer.update()
                chainer.reporter.report({'gen/loss': loss_gen})
            x_fake.unchain_backward()
            x_fake2.unchain_backward()

            critic_real = self.critic(h_real, h_fake2)
            critic_fake = self.critic(h_fake, h_fake2)

            loss_surrogate = F.mean(critic_real - critic_fake)

            eps = self.xp.random.uniform(0, 1, size=batchsize).astype("f")[
                :, None, None, None]
            x_mid = eps * x_real + (1.0 - eps) * x_fake
            grad, = chainer.grad([self.critic_process(x_mid, h_fake2.data)], [
                x_mid], enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))
            loss_gp = self.lam * \
                F.mean_squared_error(grad, xp.ones_like(grad.data))

            self.dis.cleargrads()
            (-loss_surrogate).backward()
            loss_gp.backward()
            dis_optimizer.update()

            chainer.reporter.report({'critic/loss': -loss_surrogate + loss_gp})
            chainer.reporter.report({"cramer distance": loss_surrogate})
            chainer.reporter.report({'critic/loss_grad': loss_gp})
            chainer.reporter.report({'g': F.mean(grad)})
