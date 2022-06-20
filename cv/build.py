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
    n_folds = args.n_folds

    if cv_scheme == "kfold":
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=args.random_state)
    elif cv_scheme == "gp":
        cv = GroupKFold(n_splits=n_folds)
    elif cv_scheme == "stratified":
        cv = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=args.random_state
        )
    elif cv_scheme == "gpts":
        n_time_step_oof = args.oof_size
        max_train_n_time_step = None  # args.max_train_size
        cv = GPTSSplit(
            n_splits=n_folds,
            n_time_step_oof=n_time_step_oof,
            max_train_n_time_step=max_train_n_time_step,
        )

    return cv
