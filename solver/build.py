"""
Solver building logic.
Author: JiaWei Jiang

This file contains the basic logic of building solvers for training and
evaluation processes, including the optimization algorithm and learning
rate scheduler.
"""
from typing import Any, Optional, Union

from torch import optim
from torch.nn import Module
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler


def build_optimizer(model: Module, **proc_cfg: Any) -> Optional[Optimizer]:
    """Build and return an optimizer based on configuration.

    Parameters:
        model: model instance
        proc_cfg: hyperparameters for training and evaluation processes

    Return:
        optimizer: optimization algorithm
    """
    # Setup configuration
    optim_cfg = proc_cfg["solver"]["optimizer"]
    optim_name = optim_cfg["name"]
    base_lr = float(optim_cfg["lr"])
    weight_decay = float(optim_cfg["weight_decay"])
    eps = float(optim_cfg["eps"])
    params = list(model.parameters())  # Get list of model parameters

    # Switch and initialize the specified optimizer
    optimizer: Optimizer
    if optim_name == "adadelta":
        optimizer = optim.Adadelta(
            params=params,
            rho=optim_cfg["rho"],
            eps=eps,
            lr=base_lr,
            weight_decay=weight_decay,
        )
    elif optim_name == "adagrad":
        optimizer = optim.Adagrad(
            params=params,
            lr=base_lr,
            lr_decay=optim_cfg["lr_decay"],
            weight_decay=weight_decay,
            eps=eps,
        )
    elif optim_name == "adam":
        optimizer = optim.Adam(
            params=params,
            lr=base_lr,
            betas=(optim_cfg["beta1"], optim_cfg["beta2"]),
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=optim_cfg["amsgrad"],
        )
    elif optim_name == "adamw":
        optimizer = optim.AdamW(
            params=params,
            lr=base_lr,
            betas=(optim_cfg["beta1"], optim_cfg["beta2"]),
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=optim_cfg["amsgrad"],
        )
    elif optim_name == "sgd":
        optimizer = optim.SGD(
            params=params,
            lr=base_lr,
            momentum=optim_cfg["momentum"],
            dampening=optim_cfg["dampening"],
            weight_decay=weight_decay,
            nesterov=optim_cfg["nesterov"],
        )
    else:
        raise RuntimeError(f"Optimizer {optim_name} isn't registered...")

    return optimizer


def build_lr_scheduler(
    optimizer: Optimizer, **proc_cfg: Any
) -> Optional[Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau]]:
    """Build and return a learning rate scheduler.

    Learning scheduler allows dynamic learning rate adjustment based
    on the number of epochs or monitored metric.

    Parameters:
        optimizer: optimization algorithm
        proc_cfg: hyperparameters for training and evaluation processes

    Return:
        lr_skd: learning rate scheduler
    """
    # Setup configuration
    lr_skd_cfg = proc_cfg["solver"]["lr_skd"]
    lr_skd_name = lr_skd_cfg["name"]

    # Switch and initialize the specified learning rate scheduler
    lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau]
    if lr_skd_name == "multistep":
        # Decay the learning rate of each parameter group by gamma once
        # the number of epoch reaches one of the milestones
        milestones, gamma = lr_skd_cfg["milestones"], lr_skd_cfg["gamma"]
        lr_skd = lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    elif lr_skd_name == "exp":
        # Decay the learning rate of each parameter group by gamma
        # every epoch
        gamma = lr_skd_cfg["gamma"]
        lr_skd = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)
    elif lr_skd_name == "cos":
        # Set the learning rate of each parameter group using cos
        # annealing
        T_max, eta_min = lr_skd_cfg["T_max"], lr_skd_cfg["eta_min"]
        lr_skd = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=T_max, eta_min=eta_min
        )
    elif lr_skd_name == "plateau":
        # Reduce learning rate when the monitored metric has stopped
        # improving
        factor, patience = lr_skd_cfg["factor"], lr_skd_cfg["patience"]
        lr_skd = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=factor,
            patience=patience,
        )
    elif lr_skd_name is None:
        # LR scheduler is disabled.
        print("LR scheduler is disabled...")
        return None
    else:
        raise RuntimeError(f"LR scheduler {lr_skd_name} isn't registered...")

    return lr_skd
