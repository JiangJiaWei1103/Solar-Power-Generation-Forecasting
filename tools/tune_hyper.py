"""
Main script for hyperparameter tuning.
Author: JiaWei Jiang
"""
import os
import warnings
from argparse import Namespace
from copy import deepcopy
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import early_stopping, log_evaluation
from optuna.trial import Trial
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import LabelEncoder

from config.config import gen_exp_id, setup_dp
from cv.build import build_cv
from data.data_processor import DataProcessor
from engine.defaults import BaseArgParser
from metadata import TARGET
from modeling.build import build_models
from utils.traits import is_gbdt_instance  # type: ignore

warnings.simplefilter("ignore")


class TuningArgParser(BaseArgParser):
    """Argument parser for automated hyperparameter tuning."""

    def __init__(self) -> None:
        super().__init__()

    def _build(self) -> None:
        """Build argument parser."""
        self.argparser.add_argument(
            "--input-path",
            type=str,
            default=None,
            help="path of the input file",
        )
        self.argparser.add_argument(
            "--model-name",
            type=str,
            default=None,
            help="name of the model architecture",
        )
        self.argparser.add_argument(
            "--cv-scheme", type=str, default=None, help="cross-validation scheme"
        )
        self.argparser.add_argument(
            "--n-folds", type=int, default=None, help="total number of folds"
        )
        self.argparser.add_argument(
            "--oof-size",
            type=int,
            default=None,
            help="numeber of dates in oof",
        )
        self.argparser.add_argument(
            "--stratified",
            type=str,
            default=None,
            help="column to retain class ratio",
        )
        self.argparser.add_argument(
            "--group", type=str, default=None, help="column to group CV folds"
        )
        self.argparser.add_argument(
            "--random-state",
            type=int,
            default=None,
            help="random state seeding shuffling process of cross validator",
        )
        self.argparser.add_argument(
            "--n-trials", type=int, default=None, help="number of trials run in study"
        )


def objective(
    trial: Trial,
    dp: DataProcessor,
    model_name: str,
    cv: BaseCrossValidator,
    fit_params: Dict[str, Any],
    stratified: Optional[str] = None,
    group: Optional[str] = None,
) -> float:
    """Objective function returning the value to optimize (minimize in
    this case).

    Parameters:
        trial: a single process of evaluating an objective function
        dp: data processor
        model_name: name of the estimator
        cv: cross validator
        fit_params: parameters passed to `fit()` of the estimator
        stratified: column acting as stratified determinant, used to
            preserve the percentage of samples for each class
        group: column name of group labels

    Return:
        prf: average performance of the trial
    """

    def _predict(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
        y_pred = model.predict(X)

        return y_pred

    y_, groups = _get_cv_aux(dp, stratified, group)
    X, y = dp.get_X_y()

    # Build model based on hyperparam suggestion
    trial_params = _get_trial_params(model_name, trial)
    base_model = build_models(model_name, trial_params, 1)[0]

    oof_pred = np.zeros(len(X))
    prfs = []
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_, groups)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]  # type: ignore
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]  # type: ignore

        model = clone(base_model)

        # Setup fit parameters
        fit_params_fold = deepcopy(fit_params)
        if is_gbdt_instance(model, ("lgbm", "xgb", "cat")):
            fit_params_fold["eval_set"] = [(X_tr, y_tr), (X_val, y_val)]

            if is_gbdt_instance(model, "lgbm"):
                fit_params_fold["categorical_feature"] = dp.get_cat_feats()
            elif is_gbdt_instance(model, "cat"):
                fit_params_fold["cat_features"] = dp.get_cat_feats()

        if is_gbdt_instance(model, "lgbm"):
            cb = [
                early_stopping(stopping_rounds=500, verbose=False),
                log_evaluation(period=0, show_stdv=False),
            ]
            model.fit(X_tr, y_tr, **fit_params_fold, callbacks=cb)
        else:
            model.fit(X_tr, y_tr, **fit_params_fold)

        oof_pred[val_idx] = _predict(model, X_val)
        prfs.append(np.sqrt(mse(y_val, oof_pred[val_idx])))  # Tmp hard-coded

    prf = np.mean(prfs)

    return prf


def _get_cv_aux(
    dp: DataProcessor,
    stratified: Optional[str] = None,
    group: Optional[str] = None,
) -> Tuple[Union[pd.Series, np.ndarray], Optional[Union[pd.Series, np.ndarray]]]:
    """Return auxiliary information for cv (e.g, stratified labels,
    group labels).

    Parameters:
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
        label_enc = LabelEncoder()
        y_ = label_enc.fit_transform(df[stratified])
    else:
        y_ = df[TARGET]
    groups = None if group is None else df[group]

    return y_, groups


def _get_trial_params(model_name: str, trial: Trial) -> Dict[str, Any]:
    """Return suggested hyperparameter set of the current trial.

    Parameters:
        model_name: name of the estimator
        trial: a single process of evaluating an objective function

    Return:
        model_params: parameters of the estimator
    """
    if model_name == "lgbm":
        model_params = {
            "task": "train",
            "objective": "regression",
            "boosting": "gbdt",
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.005, 0.01, 0.015, 0.02]
            ),
            "num_leaves": trial.suggest_int("num_leaves", 10, 1000),
            "num_threads": -1,
            "device": "gpu",
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8, 10, 12]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "min_sum_hessian_in_leaf": trial.suggest_loguniform(
                "min_sum_hessian_in_leaf", 0.001, 0.1
            ),
            "bagging_fraction": trial.suggest_loguniform("bagging_fraction", 0.5, 0.99),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1, 3, 5]),
            "feature_fraction": trial.suggest_loguniform("feature_fraction", 0.5, 0.99),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 0.1, 2),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 0.1, 7),
            "cat_smooth": trial.suggest_int("cat_smooth", 1, 100),
            "n_estimators": 10000,  # 2500 -> 10000
            "random_state": 42,
        }
    elif model_name == "xgb":
        model_params = {
            "booster": "gbtree",
            "verbosity": 0,
            "n_jobs": -1,
            "n_estimators": 10000,  # 2500 -> 10000
            "max_depth": trial.suggest_categorical("max_depth", [4, 6, 8, 10, 12]),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.005, 0.01, 0.015, 0.02]
            ),
            "tree_method": "auto",
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.001, 20),
            "subsample": trial.suggest_loguniform("subsample", 0.5, 0.99),
            "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.5, 0.99),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 0.1, 2),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 0.1, 7),
            "importance_type": "gain",
            "objective": "reg:squarederror",
            "random_state": 42,
        }
    elif model_name == "cat":
        # Time cost is high
        model_params = {
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [0.005, 0.01, 0.015, 0.02]
            ),
            "depth": trial.suggest_categorical("depth", [4, 6, 8, 10, 12]),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 10),
            "iterations": 10000,  # 2500 -> 10000
            "early_stopping_rounds": 500,  # 250 -> 500
            "leaf_estimation_iterations": 1,
            "eval_metric": "RMSE",
            "task_type": "CPU",
            "random_state": 16888,
        }

    return model_params


def main(args: Namespace) -> None:
    """Run automated hyperparameter tuning.

    Parameters:
        args: arguments driving automated hyperparameter tuning

    Return:
        None
    """
    input_path = args.input_path
    model_name = args.model_name

    fit_params: Dict[str, Any] = {}
    if model_name == "lgbm":
        fit_params = {"eval_metric": "rmse"}
    elif model_name == "xgb":
        fit_params = {
            "eval_metric": "rmse",
            "early_stopping_rounds": 500,
            "verbose": False,
        }
    elif model_name == "cat":
        fit_params = {
            "use_best_model": True,
            "verbose_eval": False,
        }

    # Clean and process data to feed
    dp_cfg = setup_dp()
    dp = DataProcessor(input_path, **dp_cfg)
    dp.run_before_cv()

    # Build cross validator
    cv = build_cv(args)

    # Create objective function
    objective_fn = partial(
        objective,
        dp=dp,
        model_name=model_name,
        cv=cv,
        fit_params=fit_params,
        stratified=args.stratified,
        group=args.group,
    )

    # Start automated hyperparameter tuning
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_fn, n_trials=args.n_trials)

    # Dump tuning results
    if input_path.endswith("season.csv"):
        dump_dir = "season"
    elif input_path.endswith("recency.csv"):
        dump_dir = "recency"
    else:
        dump_dir = "normal"
    dump_path = os.path.join("tuning_tmp", dump_dir, f"{gen_exp_id(model_name)}")
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)
    else:
        dump_path = os.path.join("tuning_tmp", dump_dir, f"{gen_exp_id(model_name)}")
        os.mkdir(dump_path)

    args = vars(args)
    best_prf = study.best_value
    best_params = study.best_params

    with open(os.path.join(dump_path, "tuning_args.yaml"), "w") as f:
        yaml.dump(args, f)
    open(os.path.join(dump_path, f"{best_prf}"), "a").close()
    with open(os.path.join(dump_path, "best_params.yaml"), "w") as f:
        yaml.dump(best_params, f)


if __name__ == "__main__":
    # Parse arguments
    arg_parser = TuningArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
