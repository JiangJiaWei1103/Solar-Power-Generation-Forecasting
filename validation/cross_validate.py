"""
Cross validation core logic.
Author: JiaWei Jiang

This file contains the core logic of running cross validation.
"""
import copy
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from category_encoders.utils import convert_input, convert_input_vector
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import BaseCrossValidator

import wandb
from data.data_processor import DataProcessor
from experiment.experiment import Experiment
from metadata import TARGET
from utils.traits import is_gbdt_instance  # type: ignore

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
    stratified: Optional[str] = None,
    group: Optional[str] = None,
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
        stratified: column acting as stratified determinant, used to
            preserve the percentage of samples for each class
        group: column name of group labels

    Return:
        cv_output: output of cross validatin process
    """

    oof: np.ndarray = None
    oof_scores: List[float] = []
    holdout: Optional[np.ndarray] = None
    holdout_scores: Optional[List[float]] = None
    imp: List[pd.DataFrame] = []

    def _cross_validate_inner() -> None:
        """Run inner-fold cross validation.

        Return:
            None
        """
        for ifold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_, groups)):
            # Configure cv fold-level experiment entry
            exp_fold = wandb.init(
                project=project,
                group=exp_id,
                job_type="train_eval",
                name=f"fold{ifold}",
            )

            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            X_tr, X_val, scl = dp.run_after_splitting(X_tr, X_val, ifold)

            # Setup fit parameters
            fit_params_ifold = copy.copy(fit_params)
            if is_gbdt_instance(models[0], ("lgbm", "xgb", "cat")):
                fit_params_ifold["eval_set"] = [(X_tr, y_tr), (X_val, y_val)]

                if is_gbdt_instance(models[0], "lgbm"):
                    fit_params_ifold["categorical_feature"] = dp.get_cat_feats()
                elif is_gbdt_instance(models[0], "cat"):
                    fit_params_ifold["cat_features"] = dp.get_cat_feats()

            models[ifold].fit(X_tr, y_tr, **fit_params_ifold)

            # Do inference on oof and (optional) holdout sets
            oof[val_idx] = _predict(models[ifold], X_val)
            evaluated[val_idx] = True
            #             oof_score = eval_fn(y_val, oof[val_idx])
            tr_score = np.sqrt(mse(y_tr, _predict(models[ifold], X_tr)))
            oof_score = np.sqrt(mse(y_val, oof[val_idx]))
            exp_fold.log({"train": {"rmse": tr_score}, "oof": {"rmse": oof_score}})
            oof_scores.append(oof_score)
            if holdout is not None:
                holdout[ifold] = _predict(models[ifold], X_test)
                #                 holdout_score = eval_fn(y_test, holdout[ifold])
                holdout_score = np.sqrt(mse(y_test, holdout[ifold]))
                exp_fold.log({"holdout": {"rmse": holdout_score}})
                holdout_scores.append(holdout_score)

            # Record feature importance
            if is_gbdt_instance(models[0], ("lgbm", "xgb", "cat")):
                if isinstance(X_tr, pd.DataFrame):
                    feats = X_tr.columns
                else:
                    feats = [str(i) for i in range(X_tr.shape[1])]
                imp.append(_get_feat_imp(models[ifold], feats, imp_type))

            exp_fold.finish()

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
    if _do_holdout(dp):
        for ofold in range(dp.holdout_splitter.get_n_splits()):
            test_idx = dp.holdout_splitter.get_holdout(ofold)
            train_idx = ~X.index.isin(test_idx)

            X_train = X.iloc[train_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            y_, groups = _get_cv_aux(dp, stratified, group)
            oof = np.zeros(len(X_train))
            evaluated = np.full(len(X_train), False)
            if X_test is not None:
                holdout = np.zeros((cv.get_n_splits(), len(X_test)))
                holdout_scores = []
            _cross_validate_inner()
    else:
        X_train, y_train = X, y

        y_, groups = _get_cv_aux(dp, stratified, group)
        oof = np.zeros(len(X_train))
        evaluated = np.full(len(X_train), False)
        _cross_validate_inner()

    cv_result = CVResult(oof, holdout, oof_scores, holdout_scores, imp)

    return cv_result


def _get_cv_aux(
    dp: DataProcessor,
    #     X_train: Union[pd.DataFrame, np.ndarray],
    #     y_train: Union[pd.Series, np.ndarray],
    stratified: Optional[str] = None,
    group: Optional[str] = None,
) -> Tuple[Union[pd.Series, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
    """Return auxiliary information for cv (e.g, stratified labels,
    group labels).

    Parameters:
        #X_train: training X set
        #y_train: training y set
        dp: data processor
        stratified: column acting as stratified determinant, used to
            preserve the percentage of samples for each class
        group: column name of group labels

    Return:
        y_: stratified labels
        groups: group labels
    """
    df = dp.get_df()
    if stratified is not None:
        from sklearn.preprocessing import LabelEncoder

        label_enc = LabelEncoder()
        y_ = label_enc.fit_transform(df[stratified])
    else:
        y_ = df[TARGET]
    groups = None if group is None else df[group]

    return y_, groups


def _do_holdout(dp: DataProcessor) -> bool:
    """Check if local holdout is done in the experiment.

    Parameters:
        dp: data processor

    Return:
        holdout: whether local holdout is done
    """
    holdout = False
    if dp.holdout_cfg["n_splits"] != 0:
        holdout = True

    return holdout


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

    if is_gbdt_instance(model, "lgbm"):
        feat_imp[f"importance_{imp_type}"] = model.booster_.feature_importance(imp_type)
    elif is_gbdt_instance(model, ("xgb", "cat")):
        feat_imp[f"importance_{imp_type}"] = model.feature_importances_

    return feat_imp
