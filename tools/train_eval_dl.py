"""
Main script for training and evaluation processes using DL.
Author: JiaWei Jiang

This script is used to run training and evaluation processes given the
specified CV scheme. And, evaluation can be done on unseen (testing)
data optionally.
"""
import warnings
from argparse import Namespace
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from category_encoders.utils import convert_input, convert_input_vector

import wandb
from criterion.build import build_criterion
from cv.build import build_cv
from data.build import build_dataloaders
from data.data_processor import DataProcessor
from engine.defaults import TrainEvalArgParser
from evaluating.build import build_evaluator
from experiment.experiment import Experiment
from metadata import TARGET
from modeling.dl.build import build_model
from solver.build import build_lr_scheduler, build_optimizer
from trainer.trainer import MainTrainer
from utils.common import count_params
from utils.early_stopping import EarlyStopping

warnings.simplefilter("ignore")


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
        from sklearn.preprocessing import LabelEncoder

        label_enc = LabelEncoder()
        y_ = label_enc.fit_transform(df[stratified])
    else:
        y_ = df[TARGET]
    groups = None if group is None else df[group]

    return y_, groups


def main(args: Namespace) -> None:
    """Run training and evaluation processes.

    Parameters:
        args: arguments driving training and evaluation processes

    Returns:
        None
    """
    # Configure experiment
    experiment = Experiment(args, dl=True)

    with experiment as exp:
        # Retrieve and dump configuration
        dp_cfg = exp.dp_cfg
        model_cfg = exp.model_cfg
        proc_cfg = exp.proc_cfg
        exp.dump_cfg(dp_cfg, "dp")
        exp.dump_cfg(model_cfg, args.model_name)
        exp.dump_cfg(model_cfg, "defaults")

        # Clean and process data to feed
        dp = DataProcessor(args.input_path, **dp_cfg)
        dp.run_before_cv()
        X, y = dp.get_X_y()
        X = convert_input(X)
        y = convert_input_vector(y, index=X.index)
        X_train, y_train = X, y

        # Build cross validator
        cv = build_cv(args)
        y_, groups = _get_cv_aux(dp, args.stratified, args.group)

        # Start cv process
        for ifold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_, groups)):
            # Configure cv fold-level experiment entry
            exp_fold = wandb.init(
                project=exp.args.project_name,
                group=exp.exp_id,
                job_type="train_eval",
                name=f"fold{ifold}",
            )

            # Build dataloaders
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            X_tr, X_val, scl = dp.run_after_splitting(X_tr, X_val, ifold)
            train_loader, val_loader = build_dataloaders(
                (X_tr, y_tr),
                (X_val, y_val),
                args.model_name,
                **proc_cfg["dataloader"],
                **dp_cfg["dataset"],
            )

            # Build model
            model = build_model(args.model_name, model_cfg)
            exp_fold.log({"model": {"n_params": count_params(model)}})
            model.to(proc_cfg["device"])
            wandb.watch(model, log="all", log_graph=True)

            # Build criterion
            loss_fn = build_criterion(**proc_cfg["loss_fn"])

            # Build solvers
            optimizer = build_optimizer(model, **proc_cfg)
            lr_skd = build_lr_scheduler(optimizer, **proc_cfg)

            # Build early stopping tracker
            if proc_cfg["patience"] != 0:
                es = EarlyStopping(proc_cfg["patience"], proc_cfg["mode"])
            else:
                es = None

            # Build evaluator
            evaluator = build_evaluator(proc_cfg["evaluator"]["eval_metrics"])

            # Run main training and evaluation for one fold
            trainer = MainTrainer(
                proc_cfg,
                model,
                loss_fn,
                optimizer,
                lr_skd,
                es,
                evaluator,
                train_loader,
                eval_loader=val_loader,
            )
            y_preds = trainer.train_eval(ifold)

            # Dump oof prediction
            # Different from dumping mechanism of traditional ML
            # =Think how to align validation sample=
            exp.dump_ndarr(f"oof_fold{ifold}", y_preds["oof"].numpy())

            exp_fold.finish()


if __name__ == "__main__":
    # Parse arguments
    arg_parser = TrainEvalArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
