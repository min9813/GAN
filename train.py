#!usr/bin/python
# -*- coding: UTF-8 -*-

import chainer
import math
import argparse
import sys
import os
import cupy
import time
import common.net
from chainer import training
from chainer.training import extensions
from common.utils import WeightClipping, check_and_make_dir
from common.draw import out_generated_image, gaussian_mixture_circle
from common import dataset


GRADIENT_PENALTY_WEIGHT = 10


def train(step=(5, 100),
          data_set='cifar10',
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
          gamma=0.5,
          perturb_weight=0.5,
          method="wgangp",
          techniques=None):

    check_and_make_dir(out_folder)
    new_folder = "folder_{}".format(len(os.listdir(out_folder)))
    out_folder = os.path.join(out_folder, new_folder)
    if debug:
        max_time = (1000, "iteration")
    # Make a specified GPU current

    updater_args = {"n_dis": step,
                    "device": 0}

    if data_set == 'cifar10':
        print("use cifar dataset")
        # ndim=3 : (ch,width,height)
        train = dataset.Cifar10Dataset(is_need_conditional=is_cgan)
        train_iter = chainer.iterators.SerialIterator(train, batch_size)
    elif data_set == "mnist":
        print("use mnist dataset")
        # Load the MNIST dataset
        # ndim=3 : (ch,width,height)
        train = dataset.Mnist10Dataset(is_need_conditional=is_cgan)
        train_iter = chainer.iterators.SerialIterator(
            train, batch_size, shuffle=False)
    elif data_set == "toy":
        if is_cgan:
            raise NotImplementedError
        train = gaussian_mixture_circle(60000, std=0.1)
        train_iter = chainer.iterators.SerialIterator(train, batch_size)
    else:
        sys.exit("data_set argument must be next argument [{}]".format(
            "'cifar10','mnist','toy'"))

    # Setup an optimizer
    def make_optimizer(model, **params):
        if method == "dcgan":
            # parametor require 'alpha','beta1','beta2'
            optimizer = chainer.optimizers.Adam(
                alpha=params["alpha"], beta1=params["beta1"])
            optimizer.setup(model)
            # optimizer.add_hook(
            # chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        elif method == "wgan":
            optimizer = chainer.optimizers.RMSprop(lr=params["lr"])
            optimizer.setup(model)
            try:
                optimizer.add_hook(WeightClipping(params["clip"]))
            except KeyError:
                pass
        elif method in ("sngan", "wgangp", "began", "cramer", "dragan", "improve_technique"):
            optimizer = chainer.optimizers.Adam(
                alpha=params["alpha"], beta1=params["beta1"], beta2=params["beta2"])
            optimizer.setup(model)
        else:
            raise NotImplementedError

        return optimizer

    if method == "dcgan":
        if data_set == "mnist":
            gen = common.net.Generator(n_hidden)
            dis = common.net.Discriminator()
        elif data_set == 'cifar10':
            gen = common.net.CifarGenerator(n_hidden)
            dis = common.net.CifarDiscriminator()
        else:
            gen = common.net.FCGenerator(n_hidden)
            dis = common.net.FCDiscriminator()
        from dcgan.updater import Updater
        opt_gen = make_optimizer(gen, alpha=0.0002, beta1=0.5)
        opt_dis = make_optimizer(dis, alpha=0.0002, beta1=0.5)

        plot_report = ["gen/loss", "dis/loss"]
        print_report = plot_report

    elif method == "wgan":
        if data_set == "mnist":
            gen = common.net.Generator(n_hidden)
            dis = common.net.Discriminator()
        elif data_set == 'cifar10':
            gen = common.net.CifarGenerator(n_hidden)
            dis = common.net.CifarDiscriminator()
        else:
            gen = common.net.FCGenerator(n_hidden)
            dis = common.net.FCDiscriminator()
        from wgan.updater import Updater
        opt_gen = make_optimizer(gen, lr=5e-5)
        opt_dis = make_optimizer(dis, lr=5e-5, clip=0.01)
        plot_report = ["gen/loss", 'wasserstein distance']
        print_report = plot_report

    elif method == "wgangp":
        if data_set == "mnist":
            gen = common.net.Generator(n_hidden)
            dis = common.net.Discriminator()
        elif data_set == 'cifar10':
            gen = common.net.CifarGenerator(n_hidden)
            dis = common.net.WGANDiscriminator()
        else:
            gen = common.net.FCGenerator(n_hidden)
            dis = common.net.FCDiscriminator()
        from wgangp.updater import Updater, CGANUpdater
        opt_gen = make_optimizer(gen, alpha=0.0002, beta1=0, beta2=0.9)
        opt_dis = make_optimizer(dis, alpha=0.0002, beta1=0, beta2=0.9)

        updater_args["gradient_penalty_weight"] = GRADIENT_PENALTY_WEIGHT

        plot_report = ["gen/loss", 'wasserstein distance']
        print_report = plot_report + ["critic/loss_grad", "critic/loss"]
    elif method == "began":
        import began
        from began.updater import Updater
        if data_set == "mnist":
            gen = began.net.MnistGenerator(n_hidden)
            dis = began.net.MnistDiscriminator()
        elif data_set == 'cifar10':
            gen = began.net.CifarGenerator(n_hidden)
            dis = began.net.CifarDiscriminator()
        updater_args["gamma"] = gamma
        updater_args["lambda_k"] = 0.001
        opt_gen = make_optimizer(gen, alpha=0.0002, beta1=0, beta2=0.9)
        opt_dis = make_optimizer(dis, alpha=0.0002, beta1=0, beta2=0.9)
        plot_report = ["dis/loss", "gen/loss"]
        print_report = plot_report + ["kt", "measurement"]
    elif method == "dragan":
        from dragan.updater import Updater
        if data_set == "mnist":
            gen = common.net.Generator(n_hidden)
            dis = common.net.Discriminator()
        elif data_set == 'cifar10':
            gen = common.net.CifarGenerator(n_hidden)
            dis = common.net.WGANDiscriminator()
        updater_args["gradient_penalty_weight"] = GRADIENT_PENALTY_WEIGHT
        updater_args["perturb_weight"] = perturb_weight
        opt_gen = make_optimizer(gen, alpha=0.0002, beta1=0, beta2=0.9)
        opt_dis = make_optimizer(dis, alpha=0.0002, beta1=0, beta2=0.9)
        plot_report = ["gen/loss", 'dis/loss']
        print_report = plot_report + ["dis/loss_grad"]
    elif method == "improve_technique":
        import improve_technique.net
        if data_set == "mnist":
            gen = common.net.Generator(n_hidden)
            dis = improve_technique.net.MnistMinibatchDiscriminator(use_feature_matching=techniques["feature_matching"])
        elif data_set == 'cifar10':
            gen = common.net.Discriminator(n_hidden)
            dis = improve_technique.net.CifarDeepMinibatchDiscriminator(use_feature_matching=techniques["feature_matching"])
        else:
            raise NotImplementedError
        if techniques["feature_matching"]:
            print("**feature matching**")
            from improve_technique.updater import Updater
            plot_report = ['dis/loss', "gen/loss", "gen/loss_feature"]
            print_report = plot_report
        else:
            from dcgan.updater import Updater
            plot_report = ['dis/loss', "gen/loss"]
            print_report = plot_report
        opt_gen = make_optimizer(gen, alpha=0.0002, beta1=0, beta2=0.9)
        opt_dis = make_optimizer(dis, alpha=0.0002, beta1=0, beta2=0.9)
    elif method == "sngan":
        from sngan.updater import Updater
        if data_set == "mnist":
            gen = common.net.Generator(n_hidden)
            dis = common.net.SNMnistDiscriminator()
        elif data_set == 'cifar10':
            gen = common.net.CifarGenerator(n_hidden)
            dis = common.net.SNCifarDiscriminator()
        else:
            raise NotImplementedError
        opt_gen = make_optimizer(gen, alpha=0.0002, beta1=0, beta2=0.9)
        opt_dis = make_optimizer(dis, alpha=0.0002, beta1=0, beta2=0.9)
        plot_report = ["gen/loss", 'dis/loss']
        print_report = plot_report

    else:
        raise NotImplementedError

    gen.to_gpu()  # Copy the model to the GPU
    dis.to_gpu()

    models = {"gen": gen, "dis": dis}
    opt = {"gen": opt_gen, "dis": opt_dis}
    updater_args["optimizer"] = opt
    updater_args["models"] = models
    updater_args["iterator"] = train_iter
    fixed_noise = cupy.random.uniform(-1, 1,
                                      (out_image_edge_num**2, n_hidden, 1, 1)).astype("f")
    if is_cgan:
        label_num = train.class_label_num
        updater_args["class_num"] = label_num
        updater = CGANUpdater(**updater_args)
        one_hot_label = cupy.eye(label_num)[
            cupy.arange(label_num)][:, :, None, None]
        one_hot_label = cupy.concatenate([one_hot_label] * 10)
        fixed_noise = cupy.concatenate(
            [fixed_noise, one_hot_label], axis=1).astype("f")
        print(fixed_noise.shape)

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
    if method == "began":
        trainer.extend(extensions.PlotReport(
            ["measurement"], x_key='iteration', file_name='convergence_measure.png', trigger=display_interval))
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'elapsed_time'] + print_report), trigger=display_interval)
    trainer.extend(out_generated_image(gen, out_image_edge_num, out_image_edge_num,
                                       out_image_folder, fixed_noise), trigger=(200, "iteration"))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run the training
    trainer.run()


DATASET_LIST = ["mnist", 'cifar10']
GAN_LIST = ["wgangp", "wgan", "dcgan", "cramer",
            "began", "dragan", "improve_technique", "sngan"]

parser = argparse.ArgumentParser(
    description="This file is used to train gan model")
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
                    help="max iteration time", type=int, default=1000)
parser.add_argument("-f", "--output_dir_name",
                    help="file to output the training data", default="TEST")
parser.add_argument("-l", "--latent_dim",
                    help="dimenstion of latent variable", type=int, default=128)
parser.add_argument("-m", "--method",
                    help="method to create image", choices=GAN_LIST, default="sngan")
parser.add_argument("-ndb", "--no_debug",
                    help="flag if not debug, default is False", action="store_true")
parser.add_argument("-ow", "--out_image_num",
                    help="number of output image", type=int, default=49)
parser.add_argument(
    "--cgan", help="falg to use conditional information", action="store_true")
parser.add_argument(
    "--snapshot", help="falg to save snapshot", action="store_true")

parser.add_argument("-g", "--gamma",
                    help="ratio of discriminator's output of fake image and real image, used in began to control balanced learning between two nets(for began).",
                    type=float,
                    default=0.5)
parser.add_argument("-pw", "--perturb_weight",
                    help="weight to determine what range from real data to add gradient penalty (for dragan).",
                    type=float,
                    default=0.5)
parser.add_argument("-nmd", "--no_minibatch_discrimination",
                    help="flag not to use minibatch_discrimination.",
                    action="store_true")
parser.add_argument("-fm", "--feature_matching",
                    help="flag to use feature_matching.",
                    action="store_true")

args = parser.parse_args()


def main():
    start = time.time()
    if args.cgan:
        folder_name = "cgan_" + args.method + "_" + args.dataset
        print("use {} to generate conditional image".format(args.method))
    else:
        folder_name = args.method + "_" + args.dataset
        print("use {} to generate image".format(args.method))

    out_path = os.path.join("result", folder_name,
                            "z_dim_{}".format(args.latent_dim))
    print("latent variable's dim = {}".format(args.latent_dim))
    if args.max_epoch == 100:
        stopper = (args.max_iteration, "iteration")
    else:
        stopper = (args.max_epoch, "epoch")

    assert args.gamma > 0 and args.gamma < 1, print(
        "argument 'gamma' must be between [0,1]")

    use_techniques = {"minibatch_discrimination": args.no_minibatch_discrimination is False,
                      "feature_matching":
                      args.feature_matching}

    train(step=(5, 5),
          batch_size=args.batchsize,
          data_set=args.dataset,
          debug=(args.no_debug is False),
          n_hidden=args.latent_dim,
          out_folder=out_path,
          max_time=stopper,
          save_snapshot=args.snapshot,
          out_image_edge_num=int(math.sqrt(args.out_image_num)),
          is_cgan=args.cgan,
          gamma=args.gamma,
          perturb_weight=args.perturb_weight,
          method=args.method,
          techniques=use_techniques)

    end = time.time()
    print("whole process end in {} s".format(end - start))


if __name__ == "__main__":
    main()
