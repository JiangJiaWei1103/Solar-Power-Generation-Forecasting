"""
Loss criterion building logic.
Author: JiaWei Jiang

This file contains the basic logic of building loss criterion for
training and evaluation processes.
"""
from typing import Any, Optional, Union

import torch.nn as nn
from torch.nn.modules.loss import _Loss


def build_criterion(**loss_params: Any) -> Optional[Union[_Loss]]:
    """Build and return the loss criterion.

    In some scenarios, there's a need to define loss criterion along
    with model architecture to enable the access to intermediate
    representation (e.g., multitask learning). In this case, default
    building of loss criterion is disabled and nothing is returned.

    Parameters:
        loss_params: hyperparameters for building loss function

    Return:
        criterion: loss criterion
    """
    # Setup configuration
    loss_fn = loss_params["name"]

    criterion: _Loss
    if loss_fn == "l1":
        criterion = nn.L1Loss()
    elif loss_fn == "l2":
        criterion = nn.MSELoss()
    elif loss_fn is None:
        print("Loss criterion default building is disabled...")
        return None
    else:
        raise RuntimeError(f"Loss criterion {loss_fn} isn't registered...")

    return criterion
