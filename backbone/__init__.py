# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from .GatedResNet import gatedresnet
from .TinyResNet18 import tinyresnet


def get_classifier(nclasses: int, policy, args=None) -> nn.Module:
    if policy:
        return gatedresnet(
            nclasses=nclasses,
            args=args
        )
    else:
        return tinyresnet(nclasses=nclasses, nf=args.nf)