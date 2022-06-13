"""
Script for full training.
Author: JiaWei Jiang

This script is used to train model on the whole dataset to capture
recency patterns effectively.
"""
import copy
import gc
import warnings
from argparse import Namespace
from collections import namedtuple
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error as mse

import wandb
from cv.build import build_cv
from data.data_processor import DataProcessor
from engine.defaults import TrainEvalArgParser
from experiment.experiment import Experiment
from modeling.build import build_models
from utils.traits import is_gbdt_instance  # type: ignore
from validation.cross_validate import cross_validate

warnings.simplefilter("ignore")

FullTrainResult = namedtuple("FullTrainResult", ["models", "tr_scores", "imp"])


def _full_train(
    exp: Experiment,
    dp: DataProcessor,
    model_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    best_n_estimators: int,
    imp_type: str = "gain",
) -> FullTrainResult:
    """Train model on the whole dataset.

    Parameters:
        exp: experiment logger
        dp: data processor
        model_params: parameters of the estimator
        fit_params: parameters passed to `fit()` of the estimator
        best_n_estimators: hard-coded number of boosting iterations
        imp_type: how the feature importance is calculated

    Return:
        full_train_result: output of full training process
    """

    models: List[BaseEstimator] = []
    tr_scores: List[float] = []
    imp: List[pd.DataFrame] = []

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

    # Confugure metadata
    project = exp.args.project_name
    exp_id = exp.exp_id

    # Prepare data, the whole dataset
    X, y = dp.get_X_y()
    # ===in-place scaling?===

    # Adjust model fitting process
    model_params["n_estimators"] = best_n_estimators
    model_params["early_stopping_round"] = 0

    for i, seed in enumerate([8, 168, 88, 888, 2022]):
        # Configure full training seed-level experiment entry
        exp_seed = wandb.init(
            project=project, group=exp_id, job_type="train_all", name=f"seed{i}"
        )
        model_params["random_state"] = seed
        exp_seed.log({"model": model_params})

        model = build_models(exp.args.model_name, model_params, 1)[0]

        # Setup fit parameters
        fit_params_seed = copy.copy(fit_params)
        if is_gbdt_instance(model, ("lgbm", "xgb")):
            fit_params_seed["eval_set"] = [(X, y)]

            if not is_gbdt_instance(model, "xgb"):
                fit_params_seed["categorical_feature"] = dp.get_cat_feats()

        model.fit(X, y, **fit_params_seed)
        models.append(model)

        # Do inference on the whole dataset
        tr_score = np.sqrt(mse(y, _predict(model, X)))
        exp_seed.log({"train": {"rmse": tr_score}})
        tr_scores.append(tr_score)

        # Record feature importance
        if is_gbdt_instance(model, ("lgbm", "xgb")):
            if isinstance(X, pd.DataFrame):
                feats = X.columns
            else:
                feats = [str(i) for i in range(X.shape[1])]
            imp.append(_get_feat_imp(model, feats, imp_type))

        exp_seed.finish()

    full_train_result = FullTrainResult(models, tr_scores, imp)

    return full_train_result


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
    elif is_gbdt_instance(model, "xgb"):
        feat_imp[f"importance_{imp_type}"] = model.feature_importances_

    return feat_imp


def _get_best_n_estimaters(
    cv_scheme: str, models: List[BaseEstimator]
) -> Optional[int]:
    """Derive and return number of boosted trees to fit in full
    training process.

    Parameters:
        cv_scheme: cross-validation scheme
        models: well-trained estimators
    """
    if is_gbdt_instance(models[-1], "lgbm"):
        if cv_scheme == "gpts":
            best_n_estimators = int(models[-1].best_iteration_ / 0.6)
        else:
            best_n_estimators = 0
            for model in models:
                best_n_estimators += model.best_iteration_ / len(models)
            best_n_estimators = int(best_n_estimators / (1 - 1 / len(models)))
    elif is_gbdt_instance(models[-1], "xgb"):
        if cv_scheme == "gpts":
            best_n_estimators = int(models[-1].best_iteration / 0.6)
        else:
            best_n_estimators = 0
            for model in models:
                best_n_estimators += model.best_iteration / len(models)
            best_n_estimators = int(best_n_estimators / (1 - 1 / len(models)))
    else:
        best_n_estimators = None

    return best_n_estimators


def main(args: Namespace) -> None:
    """Run training and evaluation processes, then retrain model on
    the whole training set, called full training .

    Parameters:
        args: arguments driving training, evaluation and full training
            processes

    Return:
        None
    """
    # Configure experiment
    experiment = Experiment(args)

    with experiment as exp:
        # Clean and process data to feed
        exp.dump_cfg(exp.dp_cfg, "dp")
        dp = DataProcessor(args.input_path, **exp.dp_cfg)
        dp.run_before_cv()

        # Build cross validator
        cv = build_cv(args)

        # Build models
        exp.dump_cfg(exp.model_cfg, args.model_name)
        models = build_models(args.model_name, exp.model_params, cv.get_n_splits())

        # Start cv process
        cv_result = cross_validate(
            exp=exp,
            dp=dp,
            models=models,
            cv=cv,
            fit_params=exp.fit_params,
            stratified=args.stratified,
            group=args.group,
        )

        # Dump cv results
        exp.dump_ndarr("oof", cv_result.oof_pred)
        if cv_result.holdout_pred is not None:
            for fold, holdout in enumerate(cv_result.holdout_pred):
                exp.dump_ndarr(f"holdout_fold{fold}", holdout)
        exp.incorp_meta_feats(cv_result.oof_pred)

        for fold, model in enumerate(models):
            exp.dump_model(model, model_type="fold", mid=fold)

        for fold, imp in enumerate(cv_result.imp):
            exp.dump_df(imp, f"imp/fold{fold}", ext="parquet")

        # Free cv-related objects
        del cv, cv_result
        _ = gc.collect()

        # Start full training process
        best_n_estimators = _get_best_n_estimaters(args.cv_scheme, models)
        full_train_result = _full_train(
            exp=exp,
            dp=dp,
            model_params=exp.model_params,
            fit_params=exp.fit_params,
            best_n_estimators=best_n_estimators,
        )

        # Dump full training result
        for seed, model in enumerate(full_train_result.models):
            exp.dump_model(model, model_type="whole", mid=seed)

        for seed, imp in enumerate(full_train_result.imp):
            exp.dump_df(imp, f"imp/seed{seed}", ext="parquet")


if __name__ == "__main__":
    # Parse arguments
    arg_parser = TrainEvalArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
