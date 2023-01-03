# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--nf', type=int, default=32,
                        help='number of base filters in resblocks')
    parser.add_argument('--data-path', type=str, required=True,
                        help='root path to datasets. all datasets branch out from this base path')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, required=True,
                        help='The number of epochs for each task.')


def add_policy_args(parser:ArgumentParser) -> None:
    """
    Adds the arguments used for policy / blockdrop.
    :param parser: the parser instance
    """
    parser.add_argument('--policy-penalty', type=float, default=-500,
                        help='gamma: reward for incorrect predictions')
    parser.add_argument('--warmup', action="store_true",
                        help='no actions for first 10 epochs/iterations')
    parser.add_argument('--infer-mode', type=str, default="threshold",
                        help='mode of choosing action from policy at inference',
                        choices=["sample", "threshold"])
    parser.add_argument('--policy-alpha', type=float, default=0.2,
                        help='weight given to policy logits replay')
    parser.add_argument('--reward-weight', type=float, default=0.5,
                        help='weight given to policy gradients')
    parser.add_argument('--inf-thresh', type=float, default=0.5,
                        help='probability threshold for agents during inference to keep a filter')
    parser.add_argument('--keep-ratio', type=float, default=0.7,
                        help='ratio of filters to be retained')
    parser.add_argument('--hidden-dim-ratio',
                        type=int,
                        default=16,
                        help='when using self-attention network for agents, ratio of input_dims to hidden_dims')
    parser.add_argument('--prototype-loss',
                        type=float,
                        default=0.3,
                        help='weight on prototype loss on channel-wise vectors')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.15,
                        help='sigmoid temperature for actions')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--save-path', type=str, required=True,
                        help='root path to results and models')
    parser.add_argument('--save-all', action="store_true",
                        help='save models after each non-final task')
    parser.add_argument('--exp-name', type=str, default="test",#datetime.now().strftime("%m-%d-%Y-%H-%M-%S"),
                        help='experiment name')
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, required=True,
                        help='The batch size of the memory buffer.')
