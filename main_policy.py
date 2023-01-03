# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import sys
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')

from datasets import NAMES as DATASET_NAMES
from datasets import GCL_NAMES
from models import get_all_models, policy_models
from argparse import ArgumentParser
from utils.args import add_management_args, add_policy_args
from utils.continual_training import train as ctrain
from utils.policy_continual_training import train as policy_ctrain
from datasets import get_dataset
from models import get_model
from utils.policy_training import train as policy_train
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.policy_losses import filter_loss
from backbone import get_classifier


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def main():
    lecun_fix()
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        parser.add_argument('--data-path', type=str, required=True,
                            help='Path to data')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        add_policy_args(parser)
        args = parser.parse_args()

        best = best_args[args.dataset][args.model]
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
        print(args)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    dataset = get_dataset(args)
    policy = args.model in policy_models
    if policy:
        loss = filter_loss
    else:
        loss = dataset.get_loss()
    if dataset.NAME in GCL_NAMES:
        nclasses = dataset.N_CLASSES
    else:
        nclasses = dataset.N_CLASSES_PER_TASK * dataset.N_TASKS

    backbone = get_classifier(
        nclasses=nclasses,
        policy=policy,
        args=args
    )
    model = get_model(args, backbone, loss, dataset.get_transform())

    if dataset.NAME in GCL_NAMES:
        assert not hasattr(model, 'end_task')
        if policy:
            policy_ctrain(model, dataset, args)
        else:
            ctrain(model, dataset, args)
    else:
        if policy:
            policy_train(model, dataset, args)
        else:
            train(model, dataset, args)


if __name__ == '__main__':
    main()
