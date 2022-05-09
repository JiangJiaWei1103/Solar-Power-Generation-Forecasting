"""
Experiment logger.
Author: JiaWei Jiang

This file contains the definition of experiment logger for experiment
configuration, message logging, object dumping, etc.
"""
from __future__ import annotations

import os
import pickle
from argparse import Namespace
from shutil import make_archive, rmtree
from types import TracebackType
from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

import wandb
from config.config import gen_exp_id, setup_dp, setup_model
from paths import DUMP_PATH


class Experiment(object):
    """Experiment logger.

    Parameters:
        args: arguments driving training and evaluation processes
    """

    cfg: Dict[str, Dict[str, Any]]
    model_params: Dict[str, Any]
    fit_params: Optional[Dict[str, Any]] = {}

    def __init__(self, args: Namespace):
        self.exp_id = gen_exp_id(args.model_name)
        self.args = args
        self.dp_cfg = setup_dp()
        self.model_cfg = setup_model(args.model_name)
        self._parse_model_cfg()
        self._agg_cfg()

        self._mkbuf()

    def __enter__(self) -> Experiment:
        self._run()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_inst: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._halt()

    def dump_ndarr(self, file_name: str, arr: np.ndarray) -> None:
        """Dump np.ndarray under corresponding path.

        Parameters:
            file_name: name of the file with .npy extension
            arr: array to dump

        Return:
            None
        """
        if file_name.startswith("oof"):
            dump_path = os.path.join(DUMP_PATH, "preds", "oof", file_name)
        elif file_name.startswith("holdout"):
            dump_path = os.path.join(DUMP_PATH, "preds", "holdout", file_name)
        np.save(dump_path, arr)

    def dump_df(self, df: pd.DataFrame, file_name: str, ext: str = "parquet") -> None:
        """Dump DataFrame under corresponding path.

        Support only for dumping feature importance df now.

        Parameters:
            file_name: name of the file with . extension
        """
        dump_path = os.path.join(DUMP_PATH, file_name)
        df.to_parquet(f"{dump_path}.{ext}", index=False)

    def dump_model(self, model: BaseEstimator, fold: int) -> None:
        """Dump estimator to corresponding path.

        Parameters:
            model: well-trained estimator
            fold: fold number at which the estimator is trained

        Return:
            None
        """
        dump_path = os.path.join(DUMP_PATH, "models", f"fold{fold}.pkl")
        with open(dump_path, "wb") as f:
            pickle.dump(model, f)

    def _parse_model_cfg(self) -> None:
        """Configure model parameters and parameters passed to fit
        method if they're provided.
        """
        self.model_params = self.model_cfg["model_params"]
        if self.model_cfg["fit_params"] is not None:
            self.fit_params = self.model_cfg["fit_params"]

    def _agg_cfg(self) -> None:
        """Aggregate sub configurations of different components into
        one summarized configuration.
        """
        self.cfg = {
            "common": vars(self.args),
            "dp": self.dp_cfg,
            "model": self.model_params,
            "fit": self.fit_params,
        }

    def _mkbuf(self) -> None:
        """Make local buffer for experiment output dumping."""
        if os.path.exists(DUMP_PATH):
            rmtree(DUMP_PATH)
        os.mkdir(DUMP_PATH)
        os.mkdir(os.path.join(DUMP_PATH, "models"))
        os.mkdir(os.path.join(DUMP_PATH, "trafos"))
        os.mkdir(os.path.join(DUMP_PATH, "preds"))
        for pred_type in ["oof", "holdout"]:
            os.mkdir(os.path.join(DUMP_PATH, "preds", pred_type))
        os.mkdir(os.path.join(DUMP_PATH, "imp"))

    def _run(self) -> None:
        """Start a new experiment entry and prepare local buffer."""
        self.exp_supr = wandb.init(
            project=self.args.project_name,
            config=self.cfg,
            group=self.exp_id,
            job_type="supervise",
            name="supr",
        )
        self._log_exp_metadata()
        self.exp_supr.finish()

    def _log_exp_metadata(self) -> None:
        """Log metadata of the experiment to Wandb."""
        print(f"=====Experiment {self.exp_id}=====")
        print(f"-> Model: {self.args.model_name}")
        print(f"-> CV Scheme: {self.args.cv_scheme}")
        print(f"-> Holdout Strategy: {self.dp_cfg['holdout']}")

    def _halt(self) -> None:
        # Push artifacts to remote
        dump_entry = wandb.init(
            project=self.args.project_name, group=self.exp_id, job_type="dumping"
        )
        artif = wandb.Artifact(name=self.args.model_name.upper(), type="output")
        artif.add_dir(DUMP_PATH)
        dump_entry.log_artifact(artif)
        dump_entry.finish()

        # Compress local outputs
        make_archive(f"./{self.exp_id}", "zip", root_dir="./", base_dir=DUMP_PATH)
