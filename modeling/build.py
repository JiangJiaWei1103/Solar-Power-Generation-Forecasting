"""
Estimator building logic.
Author: JiaWei Jiang

This file contains the basic logic of building estimators to train and
evaluate in different cv folds.
"""
from typing import Any, Dict, List

from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator


def build_models(
    cv: BaseCrossValidator, model_name: str, model_params: Dict[str, Any]
) -> List[BaseEstimator]:
    """Build and return estimators to train and evaluate in different
    cv folds.

    Parameter:
        cv: cross validator
        model_name: name of the estimator
        model_params: parameters of the estimator
    """
    n_folds = cv.get_n_splits()

    if model_name == "lgbm":
        model = LGBMRegressor

    models = [model(**model_params) for _ in range(n_folds)]

    return models
