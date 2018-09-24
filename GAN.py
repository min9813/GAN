# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import chainer
import random
from chainer.backends import cuda
from chainer import Chain, ChainList
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import os
from chainer.datasets import mnist


def reset_seed(xp, seed=0):
    random.seed(seed)
    xp.random.seed(seed)


def check_gpu():
    if chainer.cuda.available:
        xp = cuda.cupy
    else:
        xp = np
    return xp


class LinearBlock(Chain):

    def __init__(self, unit_size):
        w = chainer.initializers.HeNormal()
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(None, unit_size)
            self.bn = L.BatchNormalization(unit_size)

    def __call__(self, x):
        h = self.fc(x)
        h = F.leaky_relu(self.bn(h))

        return h


class Generator(ChainList):

    def __init__(self, img_cols=28, img_rows=28, img_channels=1):
        super(Generator, self).__init__(
            LinearBlock(256),
            LinearBlock(512),
            LinearBlock(1024),
            L.Linear(img_cols * img_rows * img_channels),
        )

    def __call__(self, x):
        for f in self:
            x = f(x)
        x = F.tanh(x)
        return x


class Discriminator(ChainList):

    def __init__(self, img_cols=28, img_rows=28, img_channels=1):
        super(Discriminator, self).__init__(
            LinearBlock(512),
            LinearBlock(256),
            L.Linear(None, 1),
        )
        self.flatten_size = img_cols * img_rows * img_channels

    def __call__(self, x):
        h = F.reshape(x, (-1, self.flatten_size))
        for f in self:
            h = f(h)

        return h


def train_gan(gen, dis, true_data,
              kstep=1,
              batch_size=128,
              max_epoch=10,
              latent_dim=100,
              gpu_id=0,
              save_epoch=100):
    xp = check_gpu()
    true_data = xp.array((true_data - 0.5) / 0.5, dtype=xp.float32)
    if gpu_id >= 0:
        gen.to_gpu(gpu_id)
        dis.to_gpu(gpu_id)
    o_gen = optimizers.MomentumSGD()
    o_dis = optimizers.MomentumSGD()
    o_gen.setup(gen)
    o_dis.setup(dis)

    epoch = 0
    half_batch = int(batch_size / 2)

    gen_loss = []
    dis_loss = []

    true_label = xp.array([1] * half_batch, dtype=xp.int32).reshape(-1, 1)
    false_label = xp.array([0] * half_batch, dtype=xp.int32).reshape(-1, 1)
    save_folder = "./gan_image"
    check_and_make_dir(save_folder)
    new_folder = "try_{}".format(len(os.listdir(save_folder)))
    save_folder = os.path.join(save_folder, new_folder)
    for epoch in range(max_epoch):
        sum_g_loss = xp.array(0, dtype=xp.float32)
        sum_d_loss = xp.array(0, dtype=xp.float32)
        for step in range(kstep):
            # train discriminator
            noise = xp.random.normal(
                0, 1, (half_batch, latent_dim)).astype(xp.float32)

            generated_images = gen(noise)

            idx = xp.random.randint(0, true_data.shape[0], half_batch)
            true_images = true_data[idx]

            d_predict_real = dis(true_images)
            d_loss_real = F.sigmoid_cross_entropy(d_predict_real, true_label)
            d_predict_gen = dis(generated_images)
            d_loss_gen = F.sigmoid_cross_entropy(d_predict_gen, false_label)

            d_loss = (d_loss_real + d_loss_gen) * 0.5
            sum_d_loss += d_loss.data

            dis.cleargrads()
            d_loss.backward()
            o_dis.update()

        # train generator
        noise = xp.random.normal(
            0, 1, (half_batch, latent_dim)).astype(xp.float32)
        generated_images = gen(noise)

        d_predict_gen = dis(generated_images)

        g_loss = F.sigmoid_cross_entropy(d_predict_gen, true_label)
        sum_g_loss += g_loss.data

        gen.cleargrads()
        g_loss.backward()
        o_gen.update()

        gen_loss.append(sum_g_loss)
        dis_loss.append(sum_d_loss)

        if epoch % 50 == 0:
            print("epoch:{}, generator loss:{}, discriminator loss:{}".format(
                epoch, str(sum_g_loss), str(sum_d_loss)))

        if epoch % save_epoch == 0:
            save_image(save_folder, gen, epoch, latent_dim)
    save_image(save_folder, gen, epoch, latent_dim)
    show_loss(gen_loss, dis_loss)


def check_and_make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def save_image(save_folder, gen, epoch, latent_dim):
    check_and_make_dir(save_folder)
    r, c = 5, 5
    xp = check_gpu()
    noise = xp.random.normal(0, 1, (r * c, latent_dim)).astype(xp.float32)

    gen_images = gen(noise)
    gen_images = cuda.to_cpu(gen_images.array)
    fig, axs = plt.subplots(r, c)
    gen_images = 0.5 * gen_images + 0.5
    gen_images = gen_images.reshape(-1, 28, 28, 1)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_images[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(save_folder, "epoch_{}.png".format(epoch)))
    plt.close()


def show_loss(x, y):
    plt.plot(range(len(x)), x, color="orange")
    plt.plot(range(len(x)), y, color="skyblue")

    plt.show()


if __name__ == "__main__":
    train_val, _ = mnist.get_mnist(withlabel=False, ndim=2)
    generator = Generator()
    discriminator = Discriminator()
    train_gan(generator, discriminator, train_val,
              max_epoch=15000, kstep=1, save_epoch=500)
