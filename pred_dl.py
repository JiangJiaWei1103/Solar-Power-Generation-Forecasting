"""
Main script for inference process using DL.
Author: JiaWei Jiang

This script is used to run inference process on testing set. The
well-trained models are pulled down from Wandb to do the prediction,
and final submission file is generated.
"""
import os
import pickle
import warnings
from argparse import Namespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from category_encoders.utils import convert_input
from torch.utils.data import DataLoader

import wandb
from data.data_processor import DataProcessor
from data.dataset import SinglePtDataset
from engine.defaults import InferArgParser
from metadata import SID
from modeling.dl.build import build_model

warnings.simplefilter("ignore")


def _get_test_data(
    artif_path: str, input_path: str, dp_cfg: Dict[str, Any]
) -> pd.DataFrame:
    """Generate and return test data.

    Parameters:
        artif_path: path of the pulled artifact
        input_path: path of the input file
        dp_cfg: hyperparameters of data processor

    Return:
        X_test: testing X set
    """
    # Clean and process data to feed
    dp_cfg["feats"].append(SID)
    dp = DataProcessor(input_path, **dp_cfg)
    dp.run_before_cv()
    X_test, _ = dp.get_X_y()
    X_test = convert_input(X_test)

    if dp_cfg["scale"]["type"] is not None:
        scl_path = os.path.join(artif_path, "trafos/fold2.pkl")
        with open(scl_path, "rb") as f:
            scl = pickle.load(f)
        X_test[scl.feature_names_in_] = scl.transform(X_test[scl.feature_names_in_])

    X_test.fillna(0, inplace=True)

    return X_test


def _get_models(
    artif_path: str, model_name: str, model_folds: List[int], device: torch.device
) -> List[nn.Module]:
    """Load and return well-trained models.

    Parameters:
        artif_path: path of the pulled artifact
        model_name: name of model architecture
        model_folds: fold numbers of models used to predict
        device: device used to run the inference process

    Return:
        models: well-trained model instances used to predict
    """
    model_cfg_path = os.path.join(artif_path, f"config/{model_name}.yaml")
    with open(model_cfg_path, "r") as f:
        model_cfg = yaml.full_load(f)

    model_path = os.path.join(artif_path, "models")
    models = []
    for fold, model_file in enumerate(sorted(os.listdir(model_path))):
        if fold not in model_folds:
            continue
        model = build_model(model_name, model_cfg)
        model.load_state_dict(torch.load(os.path.join(model_path, model_file)))
        model.to(device)
        models.append(model)

    return models


def _predict(
    test_loader: DataLoader, models: List[nn.Module], device: torch.device
) -> np.ndarray:
    """Do inference using well-trained models.

    Parameters:
        test_loader: testing data loader
        models: models used to do inference
        device: device used to run the inference process

    Return:
        pred: predicting results
    """
    n_models = len(models)

    y_pred = None
    for i, batch_data in enumerate(test_loader):
        # Retrieve batched raw data
        x = batch_data["X"].to(device)

        batch_y_pred = np.zeros(x.size(0))
        for model in models:
            # Forward pass
            output = model(x)
            batch_y_pred += output.detach().cpu().numpy() / n_models

        # Record batched output
        if i == 0:
            y_pred = batch_y_pred
        else:
            y_pred = np.concatenate((y_pred, batch_y_pred))

    return y_pred


def _gen_submission(pred: np.ndarray) -> None:
    """Generate final submission file.

    Parameters:
        pred: predicting results

    Return:
        None
    """
    sub = pd.DataFrame()
    sub["ID"] = np.arange(1, len(pred) + 1)
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
    device = args.device

    # Pull down cv results from Wandb
    model_name = args.model_name
    model_version = args.model_version
    artif = exp.use_artifact(f"{model_name.upper()}:v{model_version}", type="output")
    artif_path = artif.download()

    # Build testing data loader
    dp_cfg_path = os.path.join(artif_path, "config/dp.yaml")
    with open(dp_cfg_path, "r") as f:
        dp_cfg = yaml.full_load(f)
    X_test = _get_test_data(artif_path, args.input_path, dp_cfg)
    test_loader = DataLoader(
        SinglePtDataset((X_test, None), **dp_cfg["dataset"]),
        batch_size=64,
        shuffle=False,
        num_workers=8,
    )

    # Load well-trained models
    models = _get_models(artif_path, model_name, args.model_folds, device)

    # Do inference
    pred = _predict(test_loader, models, device)

    # Generate final submission
    _gen_submission(pred)

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
