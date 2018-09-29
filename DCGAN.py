import numpy as np
import chainer
from chainer.backends import cuda
from chainer import training
from chainer.training import extensions
from chainer.datasets import cifar
from PIL import Image
from legacy import check_and_make_dir
from updater import DCGANUpdater
from dcgan_model import Generator, CifarGenerator, Discriminator, CifarDiscriminator
import argparse
import sys
import os


def out_generated_image(gen, dis, rows, cols, seed, dst):
    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def make_image(trainer):
        n_images = rows * cols
        xp = gen.xp
        xp.random.seed(seed)
        z = chainer.Variable(xp.asarray(gen.make_hidden(n_images)))

        with chainer.using_config('train', False):
            x = gen(z)
        x = cuda.to_cpu(x.data)
        np.random.seed(seed)

        # gen_output_activation_func is sigmoid (0 ~ 1)
        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        # gen output_activation_func is tanh (-1 ~ 1)
#         x = np.asarray(np.clip((x+1) * 0.5 * 255, 0.0, 255.0), dtype=np.uint8)
        _, C, H, W = x.shape
        x = x.reshape((rows, cols, C, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, C))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + \
            '/image_epoch_{:0>4}.png'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image


def train(gen,
          dis,
          data_set="cifar",
          debug=True,
          n_hidden=100,
          epoch_intervel=(1, "epoch"),
          snapshot_interval=(100, "iteration"),
          dispplay_interval=(100, "iteration"),
          out_folder="./dcgan_mnist/",
          max_epoch=100):

    check_and_make_dir(out_folder)

    # Make a specified GPU current
    gen.to_gpu()  # Copy the model to the GPU
    dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    if data_set == "cifar":
        # ndim=3 : (ch,width,height)
        train, _ = cifar.get_cifar10(withlabel=False, ndim=3, scale=255.)
        train_iter = chainer.iterators.SerialIterator(train, 128)
    elif data_set == "mnist":
        # Load the MNIST dataset
        train, _ = chainer.datasets.get_mnist(
            withlabel=False, ndim=3, scale=255.)  # ndim=3 : (ch,width,height)
        train_iter = chainer.iterators.SerialIterator(train, 128)
    else:
        sys.exit("data_set argument must be next argument [{}]".format(
            "'cifar','mnist'"))

    new_folder = "folder_{}".format(len(os.listdir(out_folder)))
    out_folder = os.path.join(out_folder, new_folder)
    if debug:
        max_epoch = 10
    # Set up a trainer
    updater = DCGANUpdater(models=(gen, dis), iterator=train_iter, optimizer={
                           'gen': opt_gen, 'dis': opt_dis}, device=0)
    trainer = training.Trainer(updater, stop_trigger=(
        max_epoch, 'epoch'), out=out_folder)

    epoch_interval = (1, 'epoch')
    # snapshot_interval = (100, 'iteration')
    display_interval = (100, 'iteration')

    # trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)

    if not debug:
        trainer.extend(extensions.snapshot_object(
            gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
        trainer.extend(extensions.snapshot_object(
            dis, 'dis_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    trainer.extend(extensions.PlotReport(
        ['gen/loss', 'dis/loss'], x_key='iteration', file_name='loss.png', trigger=epoch_intervel))
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'gen/loss', 'dis/loss', 'elapsed_time']), trigger=display_interval)
    trainer.extend(out_generated_image(gen, dis, 10, 10,
                                       0, out_folder), trigger=epoch_interval)

    # Run the training
    trainer.run()


DATASET_LIST = ["mnist", "cifar"]

parser = argparse.ArgumentParser(
    description="This file is used to train model")
parser.add_argument("-d", "--dataset",
                    help="what dataset to generate",
                    choices=DATASET_LIST,
                    type=str)
parser.add_argument("-m", "--max_epoch",
                    help="max epoch time", type=int, default=100)
parser.add_argument("-f", "--output_file",
                    help="file to output the training data", type=str, default="TEST")
parser.add_argument("-l", "--latent_dim",
                    help="dimenstion of latent variable", type=int, default=100)
args = parser.parse_args()

if __name__ == "__main__":
    if args.dataset == "mnist":
        generator = Generator(args.latent_dim)
        discriminator = Discriminator()
    else:
        generator = CifarGenerator(args.latent_dim)
        discriminator = CifarDiscriminator()

    chainer.using_config("autotune", True)
    chainer.using_config("cudnn_deterministic", False)
    train(gen=generator,
          dis=discriminator,
          data_set=args.dataset,
          debug=args.debug,
          n_hidden=args.latent_dim,
          out_folder=args.output_file,
          max_epoch=args.max_epoch)
