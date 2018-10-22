import chainer
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
# import sys
from chainer.backends import cuda
# from chainer import Variable
from PIL import Image


def out_generated_image(gen, rows, cols, seed, dst, from_gaussian=False):
    @chainer.training.make_extension()
    def make_image(trainer):
        n_images = rows * cols
        xp = gen.xp
        xp.random.seed(seed)
        z = chainer.Variable(xp.asarray(gen.make_hidden(n_images)))

        with chainer.using_config('train', False):
            x = gen(z)
        x = cuda.to_cpu(x.data)
        np.random.seed(seed)

        preview_dir = '{}/preview'.format(dst)

        def save_figure(x, file_name="image"):
            file_name += "_iteration:{}.png".format(trainer.updater.iteration)
            preview_path = os.path.join(preview_dir, file_name)
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            x = x.reshape((rows * H, cols * W, C))
            if x.shape[2] == 1:
                Image.fromarray(x[:, :, 0]).save(preview_path)
            else:
                Image.fromarray(x).save(preview_path)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        if from_gaussian:
            if trainer.updater.epoch == 1:
                true_x = gaussian_mixture_circle(1000, std=0.1)
                plot_scatter(true_x, directory=preview_dir,
                             filename="true_scatter")
                plot_kde(true_x, directory=preview_dir, filename="true_kde")
            preview_filename = 'scatter_epoch_{:0>4}.png'.format(
                trainer.updater.epoch)
            plot_scatter(x, directory=preview_dir, filename=preview_filename)
            preview_filename = 'kde_epoch_{:0>4}.png'.format(
                trainer.updater.epoch)
            plot_kde(x, directory=preview_dir, filename=preview_filename)
        else:
            # gen output_activation_func is tanh (-1 ~ 1)
            x_ = np.asarray((x * 0.5 + 0.5) * 255.0, dtype=np.uint8)
            save_figure(x_, file_name="no_clip")

    return make_image


def gaussian_mixture_circle(batchsize, num_cluster=8, scale=1, std=1):
    rand_indices = np.random.randint(0, num_cluster, size=batchsize)
    base_angle = math.pi * 2 / num_cluster
    angle = rand_indices * base_angle - math.pi / 2
    mean = np.zeros((batchsize, 2), dtype=np.float32)
    mean[:, 0] = np.cos(angle) * scale
    mean[:, 1] = np.sin(angle) * scale
    return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)


def gaussian_mixture_double_circle(batchsize, num_cluster=8, scale=1, std=1):
    rand_indices = np.random.randint(0, num_cluster, size=batchsize)
    base_angle = math.pi * 2 / num_cluster
    angle = rand_indices * base_angle - math.pi / 2
    mean = np.zeros((batchsize, 2), dtype=np.float32)
    mean[:, 0] = np.cos(angle) * scale
    mean[:, 1] = np.sin(angle) * scale
    # Doubles the scale in case of even number
    even_indices = np.argwhere(rand_indices % 2 == 0)
    mean[even_indices] /= 2
    return np.random.normal(mean, std**2, (batchsize, 2)).astype(np.float32)


def plot_scatter(data, directory=None, filename="scatter", color="blue"):
    if directory is None:
        save = False
    else:
        save = True
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], s=10, marker="o",
                edgecolors="none", color=color)
    if save:
        plt.savefig("{}/{}".format(directory, filename))


def plot_kde(data, directory=None, filename="kde", color="Greens"):
    if directory is None:
        save = False
    else:
        save = True
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass
    plt.figure()
    sns.kdeplot(data[:, 0], data[:, 1], shade=True, cmap=color, n_levels=30)
#     plt.show()
    if save:
        plt.savefig("{}/{}".format(directory, filename))


if __name__ == "__main__":
    color = "blue"
    batchsize = 100
    one_gaussian = gaussian_mixture_circle(batchsize, std=0.1)
    double_gaussian = gaussian_mixture_double_circle(batchsize, std=0.1)
    plot_scatter(one_gaussian)
    plot_kde(one_gaussian)
