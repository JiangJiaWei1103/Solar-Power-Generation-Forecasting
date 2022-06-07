"""
Estimator building logic.
Author: JiaWei Jiang

This file contains the basic logic of building estimators to train and
evaluate in different cv folds.
"""
from typing import Any, Dict, List

from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def build_models(
    model_name: str,
    model_params: Dict[str, Any],
    n_models: int,
) -> List[BaseEstimator]:
    """Build and return estimators to train and evaluate in different
    cv folds or random seeds.

    Parameter:
        model_name: name of the estimator
        model_params: parameters of the estimator
        n_models: number of models to build
    """
    if model_name == "lgbm":
        model = LGBMRegressor
    elif model_name == "ridge":
        model = Ridge
    elif model_name == "xgb":
        model = XGBRegressor

    models = [model(**model_params) for _ in range(n_models)]

    return models
