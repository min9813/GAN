#!usr/bin/python
# -*- coding: UTF-8 -*-

import chainer
import math
from chainer import training
from chainer.training import extensions
from chainer.datasets import cifar
from common.utils import WeightClipping, check_and_make_dir
from common.draw import out_generated_image, gaussian_mixture_circle
from common import net
import argparse
import sys
import os
import cupy
import time


GRADIENT_PENALTY_WEIGHT = 10


def train(gen,
          dis,
          step=(5, 100),
          data_set="cifar",
          debug=True,
          n_hidden=128,
          batch_size=128,
          save_snapshot=False,
          epoch_intervel=(1, "epoch"),
          dispplay_interval=(100, "iteration"),
          out_folder="./wgan_mnist/",
          max_time=(1, "epoch"),
          out_image_edge_num=100,
          is_cgan=False,
          method="wgangp"):

    check_and_make_dir(out_folder)
    new_folder = "folder_{}".format(len(os.listdir(out_folder)))
    out_folder = os.path.join(out_folder, new_folder)
    if debug:
        max_time = (30, "iteration")
    # Make a specified GPU current
    gen.to_gpu()  # Copy the model to the GPU
    dis.to_gpu()

    models = {"gen": gen, "dis": dis}
    updater_args = {"n_dis": step,
                    "gradient_penalty_weight": GRADIENT_PENALTY_WEIGHT,
                    "device": 0}

    if data_set == "cifar":
        print("use cifar dataset")
        # ndim=3 : (ch,width,height)
        if is_cgan:
            train, _ = cifar.get_cifar10(ndim=3, scale=1.)
            label_num = 10
        else:
            train, _ = cifar.get_cifar10(withlabel=False, ndim=3, scale=1.)
            train = train * 2 - 1
        train_iter = chainer.iterators.SerialIterator(train, batch_size)
    elif data_set == "mnist":
        print("use mnist dataset")
        # Load the MNIST dataset
        # ndim=3 : (ch,width,height)
        if is_cgan:
            train, _ = chainer.datasets.get_mnist(ndim=3, scale=1.0)
            label_num = 10
        else:
            train, _ = chainer.datasets.get_mnist(
                withlabel=False, ndim=3, scale=1.)
            train = train * 2 - 1
        train_iter = chainer.iterators.SerialIterator(
            train, batch_size, shuffle=False)
    elif data_set == "toy":
        if is_cgan:
            raise NotImplementedError
        train = gaussian_mixture_circle(60000, std=0.1)
        train_iter = chainer.iterators.SerialIterator(train, batch_size)
    else:
        sys.exit("data_set argument must be next argument [{}]".format(
            "'cifar','mnist','toy'"))

    # Setup an optimizer
    def make_optimizer(model, **params):
        if method == "dcgan":
            # parametor require 'alpha','beta1','beta2'
            optimizer = chainer.optimizers.Adam(
                alpha=params["alpha"], beta1=params["beta1"])
            optimizer.setup(model)
            optimizer.add_hook(
                chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        elif method == "wgan":
            optimizer = chainer.optimizers.RMSprop(lr=params["lr"])
            optimizer.setup(model)
            try:
                optimizer.add_hook(WeightClipping(params["clip"]))
            except KeyError:
                pass
        elif method == "wgangp" or method == "cramer":
            optimizer = chainer.optimizers.Adam(
                alpha=params["alpha"], beta1=params["beta1"], beta2=params["beta2"])
            optimizer.setup(model)
        else:
            raise NotImplementedError

        return optimizer

    if method == "dcgan":
        opt_gen = make_optimizer(gen, alpha=0.0002, beta1=0.5)
        opt_dis = make_optimizer(dis, alpha=0.0002, beta1=0.5)

        plot_report = ["gen/loss", "dis/loss"]
        print_report = plot_report

    elif method == "wgan":
        opt_gen = make_optimizer(gen, lr=5e-5)
        opt_dis = make_optimizer(dis, lr=5e-5, clip=0.01)
        plot_report = ["gen/loss", 'wasserstein distance']
        print_report = plot_report

    elif method == "wgangp":
        from wgangp.updater import Updater, CGANUpdater
        opt_gen = make_optimizer(gen, alpha=0.0002, beta1=0, beta2=0.9)
        opt_dis = make_optimizer(dis, alpha=0.0002, beta1=0, beta2=0.9)

        plot_report = ["gen/loss", 'wasserstein distance']
        print_report = plot_report + ["critic/loss_grad", "critic/loss"]
    elif method == "cramer":
        from cramer.updater import Updater, CGANUpdater
        opt_gen = make_optimizer(gen, alpha=0.0002, beta1=0, beta2=0.9)
        opt_dis = make_optimizer(dis, alpha=0.0002, beta1=0, beta2=0.9)

        plot_report = ["gen/loss", 'cramer distance']
        print_report = plot_report + ["critic/loss_grad", "critic/loss"]
    else:
        raise NotImplementedError

    opt = {"gen": opt_gen, "dis": opt_dis}
    updater_args["optimizer"] = opt
    updater_args["models"] = models
    updater_args["iterator"] = train_iter
    fixed_noise = cupy.random.uniform(-1, 1,
                                      (out_image_edge_num**2, n_hidden, 1, 1)).astype("f")
    if is_cgan:
        updater_args["class_num"] = label_num
        updater = CGANUpdater(**updater_args)
        one_hot_label = cupy.eye(label_num)[
            cupy.arange(label_num)][:, :, None, None]
        one_hot_label = cupy.concatenate([one_hot_label] * 10)
        fixed_noise = cupy.concatenate(
            [fixed_noise, one_hot_label], axis=1).astype("f")
        print(fixed_noise.shape)

    else:
        updater = Updater(**updater_args)
        # Set up a trainer
    trainer = training.Trainer(updater, stop_trigger=max_time, out=out_folder)

    # epoch_interval = (1, 'epoch')
    save_snapshot_interval = (10000, "iteration")
    display_interval = (100, 'iteration')

    out_image_folder = os.path.join(out_folder, "preview")
    check_and_make_dir(out_image_folder)

    # trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    if method == "wgangp" or method == "wgan":
        trainer.extend(extensions.dump_graph('critic/loss'))
    if save_snapshot:
        trainer.extend(extensions.snapshot_object(
            gen, 'gen_epoch_{.updater.iteration}.npz'), trigger=save_snapshot_interval)
        trainer.extend(extensions.snapshot_object(
            dis, 'dis_epoch_{.updater.iteration}.npz'), trigger=save_snapshot_interval)
    trainer.extend(extensions.PlotReport(
        plot_report, x_key='iteration', file_name='loss.png', trigger=display_interval))
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'elapsed_time'] + print_report), trigger=display_interval)
    trainer.extend(out_generated_image(gen, out_image_edge_num, out_image_edge_num,
                                       out_image_folder, fixed_noise), trigger=(100, "iteration"))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run the training
    trainer.run()


DATASET_LIST = ["mnist", "cifar"]
GAN_LIST = ["wgangp", "wgan", "dcgan", "cramer"]

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
                    help="method to create image", choices=GAN_LIST, default="wgangp")
parser.add_argument("-ndb", "--no_debug",
                    help="flag if not debug, default is False", action="store_true")
parser.add_argument("-ow", "--out_image_num",
                    help="number of output image", type=int, default=100)
parser.add_argument(
    "--cgan", help="falg to use conditional information", action="store_true")
parser.add_argument(
    "--snapshot", help="falg to save snapshot", action="store_true")
args = parser.parse_args()


def main():
    start = time.time()
    print("use {} to generate image".format(args.method))
    if args.cgan:
        folder_name = "cgan_" + args.method + "_" + args.dataset
    else:
        folder_name = args.method + "_" + args.dataset

    out_path = os.path.join("result", folder_name,
                            "z_dim_{}".format(args.latent_dim))

    if args.dataset == "mnist":
        generator = net.Generator(args.latent_dim)
        discriminator = net.Discriminator()

    else:
        if args.method == "wgangp":
            discriminator = net.WGANDiscriminator()
        elif args.method=="cramer":
            discriminator = net.WGANDiscriminator(output_dim=256)
        else:
            discriminator = net.CifarDiscriminator()
        generator = net.CifarGenerator(args.latent_dim)
        print("latent variable's dim = {}".format(args.latent_dim))
    if args.max_epoch == 100:
        stopper = (args.max_iteration, "iteration")
    else:
        stopper = (args.max_epoch, "epoch")

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
          save_snapshot=args.snapshot,
          out_image_edge_num=int(math.sqrt(args.out_image_num)),
          is_cgan=args.cgan,
          method=args.method)

    end = time.time()
    print("whole process end in {} s".format(end - start))


if __name__ == "__main__":
    main()
