"""
Common layers in model architecture.
Author: JiaWei Jiang

This file contains commonly used nn layers in diversified model arch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Swish(nn.Module):
    """Activation function, swish."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        super(Swish, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward passing.

        Parameters:
            x: input variables

        Return:
            x: non-linearly transformed variables
        """
        x = x * torch.sigmoid(x)

        return x


class Mish(nn.Module):
    """Activation function, mish."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        super(Mish, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward passing.

        Parameters:
            x: input variables

        Return:
            x: non-linearly transformed variables
        """
        x = x * F.tanh(F.softplus(x))

        return x
