import torch
import torch.nn as nn


def difference(net1: nn.Module, net2: nn.Module):
    diff = 0
    for (param1, param2) in zip(net1.parameters(), net2.parameters()):
        diff += ((param1 - param2)**2).sum()
    return diff
