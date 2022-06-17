"""
Feature selectors.
Author: JiaWei Jiang

This file contains the definitions of several naive feature selectors
used to select the most salient feature subset for modeling.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error as mse

from utils.traits import is_gbdt_instance  # type: ignore


class FSPriorWrapper:
    """Feature selector wrapper supporting the incorporation of prior
    knowledge (i.e., manual feature selection).

    Parameters:
        pre_selected_feats: manually selected features
        pre_excluded_feats: manually excluded features
        fs: instance of base feature selector
    """

    def __init__(
        self,
        pre_selected_feats: List[str],
        pre_excluded_feats: List[str],
        fs: Any,
    ):
        self.pre_selected_feats = pre_selected_feats
        self.pre_excluded_feats = pre_excluded_feats
        self.fs = fs

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        fit_params: Dict[str, Any],
    ) -> Any:
        self.fs.fit(
            X=X,
            y=y,
            fit_params=fit_params,
            pre_selected_feats=self.pre_selected_feats,
            pre_excluded_feats=self.pre_excluded_feats,
        )

        return self.fs


class ForwardFS:
    """Foward feature selector.

    Parameters:
        model: instance of estimator to fit in feature selection proc
        cv_iter: cross validation iterator
        cat_feats: categorical features
        patience: if cv score doesn't improve for `patience` rounds,
            feature selection process is halted (not supported)

    Attibutes:
        best_feats_: best feature subset
        best_prf_: best performance
    """

    def __init__(
        self,
        model: BaseEstimator,
        cv_iter: Iterator[Tuple[np.ndarray, np.ndarray]],
        cat_feats: Optional[List[str]],
    ):
        self.model = model
        self.cv_iter = [_ for _ in cv_iter]
        self.cat_feats = cat_feats

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        fit_params: Dict[str, Any],
        pre_selected_feats: Optional[List[str]] = None,
        pre_excluded_feats: Optional[List[str]] = None,
    ) -> ForwardFS:
        """Start feature selection process.

        Parameters:
            X: input features
            y: target of the supervised task
            fit_params: parameters passed to `fit()` of the estimator
            pre_selected_feats: manually selected features
            pre_excluded_feats: manually excluded features
        """
        feats_not_to_search = []
        if pre_selected_feats is not None:
            feats_not_to_search += pre_selected_feats
            feats_selected = deepcopy(pre_selected_feats)
        else:
            feats_selected = []

        if pre_excluded_feats is not None:
            feats_not_to_search += pre_excluded_feats

        feats_to_search = [f for f in X.columns if f not in feats_not_to_search]

        search_round = 0
        round_prf = 1e18
        feat2prf: Dict[str, float] = {}
        self.best_prf_ = 1e18
        while True:
            print(f"Currently selected:\n{feats_selected}")
            print("=" * 25)

            feat2prf = {}
            for f in feats_to_search:
                print(f"Try feature {f}...")
                feats_selected_tmp = feats_selected + [f]
                X_ = X[feats_selected_tmp]
                prf = self._fit_single_set(X_, y, fit_params)
                print(f"Performance = {prf:.4f}")

                feat2prf[f] = prf

            feat2prf_ = pd.Series(feat2prf)
            round_prf = feat2prf_.min()  # Best performance of the current round
            feat_to_add = feat2prf_.idxmin()

            if round_prf < self.best_prf_:
                self.best_prf_ = round_prf
                feats_selected.append(feat_to_add)
                feats_to_search.remove(feat_to_add)
            else:
                break

            search_round += 1

        self.best_feats_ = feats_selected

        return self

    def _fit_single_set(
        self,
        X_: pd.DataFrame,
        y: pd.Series,
        fit_params: Dict[str, Any],
    ) -> float:
        """Fit estimator on the currently selected feature subset."""

        def _predict(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
            y_pred = model.predict(X)

            return y_pred

        feats_selected_tmp = X_.columns
        if is_gbdt_instance(self.model, ("lgbm", "cat")):
            num_feats = [f for f in feats_selected_tmp if f not in self.cat_feats]
            cat_feats = [f for f in feats_selected_tmp if f in self.cat_feats]

        oof_pred = np.zeros(len(X_))
        prfs = []
        for tr_idx, val_idx in self.cv_iter:
            X_tr, X_val = X_.iloc[tr_idx], X_.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            fit_params_fold = deepcopy(fit_params)
            if is_gbdt_instance(self.model, ("xgb", "lgbm", "cat")):
                fit_params_fold["eval_set"] = [(X_tr, y_tr), (X_val, y_val)]

                if is_gbdt_instance(self.model, "lgbm"):
                    fit_params_fold["categorical_feature"] = cat_feats
                elif is_gbdt_instance(self.model, "cat"):
                    fit_params_fold["cat_features"] = cat_feats

            model = clone(self.model)
            model.fit(X_tr, y_tr, **fit_params_fold)

            oof_pred[val_idx] = _predict(model, X_val)
            prfs.append(np.sqrt(mse(y_val, oof_pred[val_idx])))  # Tmp hard-coded

        return np.mean(prfs)
