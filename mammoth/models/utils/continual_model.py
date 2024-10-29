# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device
from .cn import *
from torch.nn import BatchNorm2d, LayerNorm, InstanceNorm2d, GroupNorm

def evaluate(nl):
    if nl == 'cn4':
        nl_fn = CN4
    elif nl == 'cn8':
        nl_fn = CN8
    elif nl == 'cn16':
        nl_fn = CN16  
    elif nl == 'cn32':
        nl_fn = CN32
    elif nl == 'cn64':
        nl_fn = CN64
    else:
        nl_fn = BatchNorm2d
    return nl_fn

class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone 
        self.loss = loss
        self.args = args
        self.transform = transform
        #self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.opt = SGD(self.parameters(), lr = self.args.lr)
        self.device = get_device()
        self.nl_fn = evaluate(str(args.nl))
        if self.nl_fn != nn.BatchNorm2d:
            self.replace_bn(self.net, 'model', self.nl_fn)
        
    def replace_bn(self, module, name, nl):
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if type(target_attr) == torch.nn.BatchNorm2d:
                new_bn = nl(target_attr)
                setattr(module, attr_str, new_bn)
        for name, icm in module.named_children():
            if type(icm) == torch.nn.BatchNorm2d:
                new_bn = nl(icm)
                setattr(module, name, new_bn)
            self.replace_bn(icm, name, self.nl_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass
