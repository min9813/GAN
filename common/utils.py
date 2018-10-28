import chainer.functions as F
import os


def check_and_make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


class WeightClipping(object):
    name = 'WeightClipping'

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, opt):
        for param in opt.target.params():
            param.data = F.clip(param.data, -self.threshold,
                                self.threshold).array
