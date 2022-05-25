"""
Main script for inference process.
Author: JiaWei Jiang

This script is used to run inference process on testing set. The
well-trained models are pulled down from Wandb to do the prediction,
and final submission file is generated.
"""
import os
import pickle
import warnings
from argparse import Namespace
from typing import List

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator

import wandb
from data.data_processor import DataProcessor
from engine.defaults import InferArgParser

warnings.simplefilter("ignore")


def _predict(dp: DataProcessor, models: List[BaseEstimator]) -> np.ndarray:
    """Do inference using well-trained models.

    Parameters:
        dp: data processor
        models: models used to do inference

    Return:
        pred: predicting results
    """
    n_folds = len(models)
    X, y = dp.get_X_y()

    pred = np.zeros(len(y))
    for model in models:
        pred += model.predict(X) / n_folds

    return pred


def _gen_submission(dp: DataProcessor, pred: np.ndarray) -> None:
    """Generate final submission file.

    Parameters:
        dp: data processor
        pred: predicting results

    Return:
        None
    """
    sub = pd.DataFrame()
    sub["ID"] = dp.get_df()["ID"]
    sub["Generation"] = pred
    sub.sort_values("ID", inplace=True)
    sub["ID"] = sub["ID"].astype("int64")
    sub.to_csv("submission.csv", index=False)


def main(args: Namespace) -> None:
    """Run inference process and generate submission file.

    Parameters:
        args: arguments driving inference process

    Return:
        None
    """
    # Configure experiment
    exp = wandb.init(
        project=args.project_name,
        config={"common": vars(args)},
        group=args.exp_id,
        job_type="infer",
    )

    # Pull down cv results from Wandb
    model_name = args.model_name
    model_version = args.model_version
    artif = exp.use_artifact(f"{model_name.upper()}:v{model_version}", type="output")
    artif_path = artif.download()

    # Clean and process data to feed
    dp_cfg_path = os.path.join(artif_path, "config/dp.yaml")
    with open(dp_cfg_path, "r") as f:
        dp_cfg = yaml.full_load(f)
    dp = DataProcessor(args.input_path, **dp_cfg)
    dp.run_before_cv()

    # Load well-trained models
    model_folds = args.model_folds
    model_path = os.path.join(artif_path, "models")

    models = []
    for fold, model_file in enumerate(sorted(os.listdir(model_path))):
        if fold not in model_folds:
            continue
        with open(os.path.join(model_path, model_file), "rb") as f:
            models.append(pickle.load(f))

    # Do inference
    pred = _predict(dp, models)

    # Generate final submission
    _gen_submission(dp, pred)

    # Push artifacts to remote
    artif = wandb.Artifact(name=f"{model_name.upper()}_infer", type="output")
    artif.add_file("./submission.csv")
    exp.log_artifact(artif)
    exp.finish()


if __name__ == "__main__":
    # Parse arguments
    arg_parser = InferArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
