# import os

import numpy as np
# from PIL import Image
import chainer
from chainer.dataset import dataset_mixin


class Cifar10Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, test=False, is_need_conditional=False):
        self.class_label_num = None
        if is_need_conditional:
            d_train, d_test = chainer.datasets.get_cifar10(ndim=3, scale=1.0)
        else:
            d_train, d_test = chainer.datasets.get_cifar10(
                ndim=3, withlabel=False, scale=1.0)
        if test:
            self.ims = d_test
        else:
            self.ims = d_train
        if is_need_conditional:
            tmp_ims, tmp_label = zip(*self.ims)
            tmp_ims = np.array(tmp_ims) * 2 - 1
            self.class_label_num = len(set(tmp_label))
            self.ims = list(zip(tmp_ims, tmp_label))
            print("load cifar-10.  shape: ", np.array(tmp_ims).shape)
        else:
            self.ims = self.ims * 2 - 1.0  # [-1.0, 1.0]
            print("load cifar-10.  shape: ", self.ims.shape)

    def __len__(self):
        return len(self.ims)

    def get_example(self, i):
        return self.ims[i]


class Mnist10Dataset(dataset_mixin.DatasetMixin):
    def __init__(self, test=False, is_need_conditional=False):
        self.class_label_num = None
        if is_need_conditional:
            d_train, d_test = chainer.datasets.get_mnist(ndim=3, scale=1.0)
        else:
            d_train, d_test = chainer.datasets.get_mnist(
                ndim=3, withlabel=False, scale=1.0)
        if test:
            self.ims = d_test
        else:
            self.ims = d_train
        if is_need_conditional:
            tmp_ims, tmp_label = zip(*self.ims)
            self.class_label_num = len(set(tmp_label))
            tmp_ims = np.array(tmp_ims) * 2 - 1
            self.ims = list(zip(tmp_ims, tmp_label))
            print("load cifar-10.  shape: ", np.array(tmp_ims).shape)
        else:
            self.ims = self.ims * 2 - 1.0  # [-1.0, 1.0]
            print("load mnist  shape: ", self.ims.shape)

    def __len__(self):
        return len(self.ims)

    def get_example(self, i):
        return self.ims[i]
