"""
Cross-validator building logic.
Author: JiaWei Jiang

This file contains the basic logic of building cv iterator for training
and evaluation processes.
"""
from argparse import Namespace

from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    KFold,
    StratifiedKFold,
)

from .ts import GPTSSplit


def build_cv(args: Namespace) -> BaseCrossValidator:
    """Build and return the cross validator.

    Parameters:
        args: arguments driving training and evaluation processes

    Return:
        cv: cross validator
    """
    cv_scheme = args.cv_scheme
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    n_folds = args.n_folds
    n_time_step_oof = args.oof_size
    max_train_n_time_step = args.max_train_size
    group = args.group
    random_state = args.random_state

    if cv_scheme == "kfold":
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif cv_scheme == "gp":
        cv = GroupKFold(n_splits=n_folds)
    elif cv_scheme == "stratified":
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    elif cv_scheme == "gpts":
        cv = GPTSSplit(
            n_splits=n_folds,
            n_time_step_oof=n_time_step_oof,
            max_train_n_time_step=max_train_n_time_step,
        )

    return cv
