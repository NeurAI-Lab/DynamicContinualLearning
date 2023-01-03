import torch.nn as nn
from torch.nn.functional import cross_entropy


def filter_loss(outputs, labels, rewards=None, log_probs=None):
    """
    :param outputs:
    :param labels:
    :param rewards:
    :param log_probs:
    :return: policy loss i.e -log_probs * rewards + cross entropy
    """
    if rewards is not None:
        policy_loss = (-log_probs * rewards).mean()
        main_loss = nn.functional.cross_entropy(outputs, labels)
        return main_loss, policy_loss
    else:
        main_loss = nn.functional.cross_entropy(outputs, labels)
        return main_loss
