"""
Evaluator definition.
Author: JiaWei Jiang

This file contains the definition of evaluator used during evaluation
process.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch import Tensor


class Evaluator(object):
    """Custom evaluator.

    Following is a simple illustration of evaluator used in regression
    task.

    Parameters:
        metric_names: evaluation metrics
    """

    eval_metrics: Dict[str, Callable[..., Union[float]]]

    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.eval_metrics = {}
        self._build()

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        scaler: Optional[object] = None,
        tids: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Run evaluation using pre-specified metrics.

        Parameters:
            y_true: groundtruths
            y_pred: predicting values
            scaler: scaling object
                *Note: For fair comparisons among experiments using
                    models trained on y with different scales, the
                    inverse tranformation is needed.
            tids: time identifiers

        Return:
            eval_result: evaluation performance report
        """
        if scaler is not None:
            # Do inverse transformation to rescale y values
            y_pred, y_true = self._rescale_y(y_pred, y_true, scaler)

        eval_result = {}
        for metric_name, metric in self.eval_metrics.items():
            if metric_name == "corr_t":
                eval_result[metric_name] = metric(tids, y_pred, y_true)
            else:
                eval_result[metric_name] = metric(y_pred, y_true)

        return eval_result

    def _build(self) -> None:
        """Build evaluation metric instances."""
        for metric_name in self.metric_names:
            if metric_name == "rmse":
                self.eval_metrics[metric_name] = self._RMSE
            elif metric_name == "mae":
                self.eval_metrics[metric_name] = self._MAE
            elif metric_name == "rrse":
                self.eval_metrics[metric_name] = self._RRSE
            elif metric_name == "rae":
                self.eval_metrics[metric_name] = self._RAE
            elif metric_name == "corr_t":
                self.eval_metrics[metric_name] = self._CORR_T
            elif metric_name == "corr":
                self.eval_metrics[metric_name] = self._CORR

    def _rescale_y(
        self, y_pred: Tensor, y_true: Tensor, scaler: Any  # Temporary workaround
    ) -> Tuple[Tensor, Tensor]:
        """Rescale y to the original scale.

        Parameters:
            y_pred: predicting results
            y_true: groundtruths
            scaler: scaling object

        Return:
            y_pred: rescaled predicting results
            y_true: rescaled groundtruths
        """
        n_samples = len(y_pred)
        y_pred = scaler.inverse_transform(y_pred.reshape(n_samples, -1))
        y_true = scaler.inverse_transform(y_true.reshape(n_samples, -1))

        y_pred = torch.tensor(y_pred.squeeze(), dtype=torch.float32)
        y_true = torch.tensor(y_true.squeeze(), dtype=torch.float32)

        return y_pred, y_true

    def _RMSE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Root mean squared error.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            rmse: root mean squared error
        """
        mse = nn.MSELoss()
        rmse = torch.sqrt(mse(y_pred, y_true)).item()

        return rmse

    def _MAE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Mean absolutes error.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            mae: root mean squared error
        """
        mae = nn.L1Loss()(y_pred, y_true).item()

        return mae

    def _RRSE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Root relative squared error.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            rrse: root relative squared error
        """
        gt_mean = torch.mean(y_true)

        sse = nn.MSELoss(reduction="sum")  # Sum squared error
        rrse = torch.sqrt(
            sse(y_pred, y_true) / sse(gt_mean.expand(y_true.shape), y_true)
        ).item()

        return rrse

    def _RAE(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Relative absolute error.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            rae: relative absolute error
        """
        gt_mean = torch.mean(y_true)

        sae = nn.L1Loss(reduction="sum")  # Sum absolute error
        rae = (sae(y_pred, y_true) / sae(gt_mean.expand(y_true.shape), y_true)).item()

        return rae

    def _CORR(self, y_pred: Tensor, y_true: Tensor) -> float:
        """Empirical correlation coefficient.

        Because there are some time series with zero values across the
        specified dataset (e.g., time series idx 182 in electricity
        across val and test set with size of splitting 6:2:2), corr of
        such series are dropped to avoid situations like +-inf or NaN.

        Parameters:
            y_pred: predicting results
            y_true: groudtruths

        Return:
            corr: empirical correlation coefficient
        """
        pred_mean = torch.mean(y_pred, dim=0)
        pred_std = torch.std(y_pred, dim=0)
        gt_mean = torch.mean(y_true, dim=0)
        gt_std = torch.std(y_true, dim=0)

        # Extract legitimate time series index with non-zero std to
        # avoid situations stated in *Note.
        pred_idx_leg = pred_std != 0
        gt_idx_leg = gt_std != 0
        idx_leg = torch.logical_and(pred_idx_leg, gt_idx_leg)

        corr_per_ts = torch.mean(((y_pred - pred_mean) * (y_true - gt_mean)), dim=0) / (
            pred_std * gt_std
        )
        corr = torch.mean(corr_per_ts[idx_leg]).item()  # Take mean across time series

        return corr

    def _CORR_T(self, tids: Tensor, y_pred: Tensor, y_true: Tensor) -> float:
        """Empirical correlation coefficient.

        Parameters:
            tids: time identifiers
            y_pred: predicting results
            y_true: groudtruths

        Return:
            corr: correlation coefficient
        """
        data = {
            "time_id": tids.numpy(),
            "y_pred": y_pred.numpy(),
            "y_true": y_true.numpy(),
        }
        df = pd.DataFrame.from_dict(data)
        corr = (
            df.groupby("time_id")
            .apply(lambda x: pearsonr(x["y_true"], x["y_pred"])[0])
            .mean()
            .item()
        )  # Convert np valeus to a native python type

        return corr
