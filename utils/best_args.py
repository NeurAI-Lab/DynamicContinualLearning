# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

best_args = {
    'seq-mnist': {
        'dynamos': {500: {'lr': 0.07,
                        'minibatch_size': 10,
                        'alpha': 0.2,
                        'beta': 2.0,
                        'batch_size': 10,
                        'nf': 32,
                        'warmup': True,
                        'policy_alpha': 0.2,
                        'policy_penalty': -500,
                        'reward_weight': 0.5,
                        'prototype_loss': 0.3,
                        'keep_ratio': 0.7,
                        'n_epochs': 1},
                  1000: {'lr': 0.07,
                        'minibatch_size': 10,
                        'alpha': 0.1,
                        'beta': 2.5,
                        'batch_size': 10,
                        'nf': 32,
                        'warmup': True,
                        'policy_alpha': 0.2,
                        'policy_penalty': -200,
                        'reward_weight': 0.7,
                        'prototype_loss': 0.5,
                        'keep_ratio': 0.7,
                        'n_epochs': 1},
                  2000: {'lr': 0.07,
                         'minibatch_size': 10,
                         'alpha': 0.5,
                         'beta': 3.0,
                         'batch_size': 10,
                         'nf': 32,
                         'warmup': True,
                         'policy_alpha': 0.2,
                         'policy_penalty': -200,
                         'reward_weight': 0.5,
                         'prototype_loss': 0.5,
                         'keep_ratio': 0.7,
                         'n_epochs': 1}}},
    'seq-svhn': {
        'dynamos': {500: {'lr': 0.07,
                          'minibatch_size': 16,
                          'alpha': 2.0,
                          'beta': 3.0,
                          'batch_size': 16,
                          'nf': 32,
                          'warmup': True,
                          'policy_alpha': 1.0,
                          'policy_penalty': -500,
                          'reward_weight': 1.0,
                          'prototype_loss': 0.5,
                          'keep_ratio': 0.7,
                          'n_epochs': 70},
                    1000: {'lr': 0.07,
                           'minibatch_size': 16,
                           'alpha': 2.5,
                           'beta': 2.0,
                           'batch_size': 16,
                           'nf': 32,
                           'warmup': True,
                           'policy_alpha': 0.2,
                           'policy_penalty': -500,
                           'reward_weight': 1.0,                           'prototype_loss': 0.5,
                           'keep_ratio': 0.7,
                           'n_epochs': 70},
                    2000: {'lr': 0.07,
                           'minibatch_size': 16,
                           'alpha': 2.5,
                           'beta': 2.0,
                           'batch_size': 16,
                           'nf': 32,
                           'warmup': True,
                           'policy_alpha': 0.2,
                           'policy_penalty': -500,
                           'reward_weight': 1.0,
                           'prototype_loss': 0.5,
                           'keep_ratio': 0.7,
                           'n_epochs': 70}}},
    'mnist-360': {
        'dynamos': {100: {'lr': 0.07,
                        'minibatch_size': 16,
                        'alpha': 0.2,
                        'beta': 1.0,
                        'batch_size': 16,
                        'nf': 32,
                        'warmup': True,
                        'policy_alpha': 0.1,
                        'policy_penalty': -200,
                        'reward_weight': 0.5,
                        'prototype_loss': 0.5,
                        'keep_ratio': 0.7,
                        'n_epochs': 1},
                  200: {'lr': 0.07,
                        'minibatch_size': 16,
                        'alpha': 0.2,
                        'beta': 1.5,
                        'batch_size': 16,
                        'nf': 32,
                        'warmup': True,
                        'policy_alpha': 0.1,
                        'policy_penalty': -200,
                        'reward_weight': 1.0,
                        'prototype_loss': 0.5,
                        'keep_ratio': 0.7,
                        'n_epochs': 1},
                  500: {'lr': 0.07,
                         'minibatch_size': 16,
                         'alpha': 0.1,
                         'beta': 1.5,
                         'batch_size': 16,
                         'nf': 32,
                         'warmup': True,
                         'policy_alpha': 0.1,
                         'policy_penalty': -200,
                         'reward_weight': 0.3,
                         'prototype_loss': 0.3,
                         'keep_ratio': 0.7,
                         'n_epochs': 1}}}
}
