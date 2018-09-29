import chainer
import chainer.functions as F


class DCGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
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
        x_real = chainer.Variable(self.converter(batch, self.device)) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)
        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        y_real = dis(x_real)
        z = gen.make_hidden(batchsize)
        x_fake = gen(z)
        y_fake = dis(x_fake)

        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real, xp)
        gen_optimizer.update(self.loss_gen, gen, y_fake, xp)
