import torch.nn as nn


def nll_loss(output, target, weight):

    return nn.NLLLoss(weight=weight)
