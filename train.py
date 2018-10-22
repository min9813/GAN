#!usr/bin/python
# -*- coding: UTF-8 -*-

import chainer
import math
from chainer import training
from chainer.training import extensions
from chainer.datasets import cifar
from draw import out_generated_image, gaussian_mixture_circle
from legacy import check_and_make_dir
from updater import WGANUpdater, DCGANUpdater, WGANGPUpdater
from model import Generator, CifarGenerator, Discriminator, CifarDiscriminator, WGANDiscriminator, WeightClipping
import argparse
import sys
import os


GRADIENT_PENALTY_WEIGHT = 10


def train(gen,
          dis,
          step=(5, 100),
          data_set="cifar",
          debug=True,
          n_hidden=100,
          batch_size=128,
          save_snapshot=False,
          epoch_intervel=(1, "epoch"),
          dispplay_interval=(100, "iteration"),
          out_folder="./wgan_mnist/",
          max_time=(1, "epoch"),
          out_image_edge_num=100,
          method="WGAN"):

    check_and_make_dir(out_folder)
    new_folder = "folder_{}".format(len(os.listdir(out_folder)))
    out_folder = os.path.join(out_folder, new_folder)
    if debug:
        max_time = (3, "epoch")
    # Make a specified GPU current
    gen.to_gpu()  # Copy the model to the GPU
    dis.to_gpu()

    if data_set == "cifar":
        # ndim=3 : (ch,width,height)
        train, _ = cifar.get_cifar10(withlabel=False, ndim=3, scale=1.)
        train = train * 2 - 1
        train_iter = chainer.iterators.SerialIterator(train, batch_size)
    elif data_set == "mnist":
        # Load the MNIST dataset
        # ndim=3 : (ch,width,height)
        train, _ = chainer.datasets.get_mnist(
            withlabel=False, ndim=3, scale=1.)
        train = train * 2 - 1
        train_iter = chainer.iterators.SerialIterator(train, batch_size)
    elif data_set == "toy":
        train = gaussian_mixture_circle(60000, std=0.1)
        train_iter = chainer.iterators.SerialIterator(train, batch_size)
    else:
        sys.exit("data_set argument must be next argument [{}]".format(
            "'cifar','mnist','toy'"))

    # Setup an optimizer
    def make_optimizer(model, **params):
        if method == "DCGAN":
            # parametor require 'alpha','beta1','beta2'
            optimizer = chainer.optimizers.Adam(
                alpha=params["alpha"], beta1=params["beta1"])
            optimizer.setup(model)
            optimizer.add_hook(
                chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        elif method == "WGAN":
            optimizer = chainer.optimizers.RMSprop(lr=params["lr"])
            optimizer.setup(model)
            try:
                optimizer.add_hook(WeightClipping(params["clip"]))
            except KeyError:
                pass
        elif method == "WGANGP":
            optimizer = chainer.optimizers.Adam(
                alpha=params["alpha"], beta1=params["beta1"], beta2=params["beta2"])
            optimizer.setup(model)

        return optimizer

    if method == "DCGAN":
        opt_gen = make_optimizer(gen, alpha=0.0002, beta1=0.5)
        opt_dis = make_optimizer(dis, alpha=0.0002, beta1=0.5)
        updater = DCGANUpdater(models=(gen, dis),
                               iterator=train_iter,
                               optimizer={"gen": opt_gen, "dis": opt_dis},
                               device=0)
        plot_report = ["gen/loss", "dis/loss"]
        print_report = plot_report

    elif method == "WGAN":
        opt_gen = make_optimizer(gen, lr=5e-5)
        opt_dis = make_optimizer(dis, lr=5e-5, clip=0.01)
        updater = WGANUpdater(models=(gen, dis),
                              iterator=train_iter,
                              optimizer={"gen": opt_gen, "critic": opt_dis},
                              step=step,
                              device=0)
        plot_report = ["gen/loss", 'wasserstein distance']
        print_report = plot_report

    elif method == "WGANGP":
        opt_gen = make_optimizer(gen, alpha=0.0001, beta1=0.5, beta2=0.9)
        opt_dis = make_optimizer(dis, alpha=0.0001, beta1=0.5, beta2=0.9)
        updater = WGANGPUpdater(models=(gen, dis),
                                iterator=train_iter,
                                optimizer={"gen": opt_gen, "critic": opt_dis},
                                step=step,
                                gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT,
                                device=0)
        plot_report = ["gen/loss", 'wasserstein distance']
        print_report = plot_report + ["critic/loss_grad", "critic/loss"]

    # Set up a trainer
    trainer = training.Trainer(updater, stop_trigger=max_time, out=out_folder)

    epoch_interval = (1, 'epoch')
    display_interval = (100, 'iteration')

    # trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    if method == "WGAN" or method == "WGANGP":
        trainer.extend(extensions.dump_graph('critic/loss'))
    if save_snapshot:
        trainer.extend(extensions.snapshot_object(
            gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
        trainer.extend(extensions.snapshot_object(
            dis, 'dis_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
    trainer.extend(extensions.PlotReport(
        plot_report, x_key='iteration', file_name='loss.png', trigger=display_interval))
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'elapsed_time'] + print_report), trigger=epoch_interval)
    trainer.extend(out_generated_image(gen, out_image_edge_num, out_image_edge_num,
                                       0, out_folder), trigger=(200, "iteration"))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run the training
    trainer.run()


DATASET_LIST = ["mnist", "cifar"]
GAN_LIST = ["WGANGP", "WGAN", "DCGAN"]

parser = argparse.ArgumentParser(
    description="This file is used to train model")
parser.add_argument("-d", "--dataset",
                    help="what dataset to generate",
                    choices=DATASET_LIST,
                    type=str,
                    default="mnist")
parser.add_argument("-b", "--batchsize",
                    help="batch size", type=int, default=64)
parser.add_argument("-me", "--max_epoch",
                    help="max epoch time", type=int, default=100)
parser.add_argument("-mi", "--max_iteration",
                    help="max iteration time", type=int, default=100)
parser.add_argument("-f", "--output_dir_name",
                    help="file to output the training data", type=str, default="TEST")
parser.add_argument("-l", "--latent_dim",
                    help="dimenstion of latent variable", type=int, default=100)
parser.add_argument("-m", "--method",
                    help="method to create image", choices=GAN_LIST, default="WGANGP")
parser.add_argument("-ndb", "--no_debug",
                    help="flag if not debug, default is False", action="store_true")
parser.add_argument("-ow", "--out_image_num",
                    help="number of output image", type=int, default=100)
args = parser.parse_args()


def main():
    gan_type = args.method.upper()
    print("use {} to generate image".format(args.method))
    out_path = os.path.join(gan_type.lower() + "_" + args.dataset,
                            "z_dim_{}".format(args.latent_dim))
    if args.dataset == "mnist":
        generator = Generator(args.latent_dim)
        discriminator = Discriminator()

    else:
        if args.method == "WGANGP":
            discriminator = WGANDiscriminator()
        else:
            discriminator = CifarDiscriminator()
        generator = CifarGenerator(args.latent_dim)
    if args.max_epoch == 100:
        stopper = (args.max_iteration, "iteration")
    else:
        stopper = (args.max_epoch, "epoch")
    check_and_make_dir(out_path)

    chainer.using_config("autotune", True)
    chainer.using_config("cudnn_deterministic", False)
    train(gen=generator,
          dis=discriminator,
          step=(5, 5),
          batch_size=args.batchsize,
          data_set=args.dataset,
          debug=(args.no_debug is False),
          n_hidden=args.latent_dim,
          out_folder=out_path,
          max_time=stopper,
          out_image_edge_num=int(math.sqrt(args.out_image_num)),
          method=args.method)


if __name__ == "__main__":
    main()
