# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from datasets.seq_mnist import SequentialMNIST
from datasets.mnist_360 import MNIST360
from datasets.seq_svhn import SequentialSVHN
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace


NAMES = {
    MNIST360.NAME: MNIST360,
    SequentialMNIST.NAME: SequentialMNIST,
    SequentialSVHN.NAME: SequentialSVHN
}

GCL_NAMES = {
    MNIST360.NAME: MNIST360,
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
