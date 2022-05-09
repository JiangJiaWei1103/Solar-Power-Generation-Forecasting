"""
Cross validation core logic.
Author: JiaWei Jiang

This file contains the core logic of running cross validation.
"""
import copy
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from category_encoders.utils import convert_input, convert_input_vector
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import BaseCrossValidator

import wandb
from data.data_processor import DataProcessor
from experiment.experiment import Experiment

CVResult = namedtuple(
    "CVResult", ["oof_pred", "holdout_pred", "oof_scores", "holdout_scores", "imp"]
)


def cross_validate(
    exp: Experiment,
    dp: DataProcessor,
    models: List[BaseEstimator],
    cv: BaseCrossValidator,
    fit_params: Dict[str, Any],
    eval_fn: Optional[Callable] = None,
    imp_type: str = "gain",
) -> CVResult:
    """Run cross validation and return evaluated performance and
    predicting results.

    The implementation only supports single holdout set now. The nested
    cv scheme will be implemented in the future.

    Parameters:
        exp: experiment logger
        dp: data processor
        models: list of instances of estimator to train and evaluate
        cv: cross validator
        fit_params: parameters passed to `fit()` of the estimator
        eval_fn: evaluation function used to derive performance score
        imp_type: how the feature importance is calculated

    Return:
        cv_output: output of cross validatin process
    """

    def _predict(model: BaseEstimator, x: pd.DataFrame) -> np.ndarray:
        """Do inference with the well-trained estimator.

        Parameters:
            model: well-trained estimator used to do inference
            x: data to predict on

        Return:
            y_pred: predicting results
        """
        y_pred = model.predict(x)

        return y_pred

    # Configure metadata
    project = exp.args.project_name
    exp_id = exp.exp_id

    # Process X and y sets
    X, y = dp.get_X_y()
    X = convert_input(X)
    y = convert_input_vector(y, index=X.index)

    # Start cv process
    for ofold in range(dp.holdout_splitter.get_n_splits()):
        test_idx = dp.holdout_splitter.get_holdout(ofold)
        train_idx = ~X.index.isin(test_idx)

        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        oof = np.zeros(len(X_train))
        evaluated = np.full(len(X_train), False)
        oof_scores = []
        holdout = None
        if X_test is not None:
            holdout = np.zeros((cv.get_n_splits(), len(X_test)))
            holdout_scores = []
        imp = []

        for ifold, (tr_idx, val_idx) in enumerate(
            cv.split(
                X_train,
                y_train,
            )
        ):
            # Configure cv fold-level experiment entry
            exp_fold = wandb.init(
                project=project,
                group=exp_id,
                job_type="train_eval",
                name=f"fold{ifold}",
            )

            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

            fit_params_ifold = copy.copy(fit_params)
            fit_params_ifold["eval_set"] = [(X_tr, y_tr), (X_val, y_val)]

            models[ifold].fit(X_tr, y_tr, **fit_params_ifold)

            # Do inference on oof and (optional) holdout sets
            oof[val_idx] = _predict(models[ifold], X_val)
            evaluated[val_idx] = True
            #             oof_score = eval_fn(y_val, oof[val_idx])
            oof_score = np.sqrt(mse(y_val, oof[val_idx]))
            exp_fold.log({"oof": {"rmse": oof_score}})
            oof_scores.append(oof_score)
            if holdout is not None:
                holdout[ifold] = _predict(models[ifold], X_test)
                #                 holdout_score = eval_fn(y_test, holdout[ifold])
                holdout_score = np.sqrt(mse(y_test, holdout[ifold]))
                exp_fold.log({"holdout": {"rmse": holdout_score}})
                holdout_scores.append(holdout_score)

            # Record feature importance
            imp.append(_get_feat_imp(models[ifold], X_tr.columns, imp_type))

            exp_fold.finish()

    cv_result = CVResult(oof, holdout, oof_scores, holdout_scores, imp)

    return cv_result


def _get_feat_imp(
    model: BaseEstimator, feat_names: List[str], imp_type: str
) -> pd.DataFrame:
    """Generate and return feature importance DataFrame.

    Parameters:
        model: well-trained estimator
        feat_names: list of feature names
        imp_type: how the feature importance is calculated

    Return:
        feat_imp: feature importance
    """
    feat_imp = pd.DataFrame(feat_names, columns=["feature"])
    feat_imp[f"importance_{imp_type}"] = model.booster_.feature_importance(imp_type)

    return feat_imp
