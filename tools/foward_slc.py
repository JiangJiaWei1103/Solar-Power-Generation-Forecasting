"""
Randomly sequential forward selection.
Author: JiaWei Jiang

This script is used to randomly select feature subset to feed into the
designated ML models afterward. The time cost is high, so it's not
suitable for the scenario with large #samples or #features.
"""
import os
import pickle
import warnings
from argparse import Namespace

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from config.config import gen_exp_id
from cv.ts import GPTSSplit
from data.fe import FE
from data.fs import ForwardFS, FSPriorWrapper
from engine.defaults import BaseArgParser
from metadata import PK, TARGET, TID
from modeling.build import build_models

warnings.simplefilter("ignore")

LABEL_ENC_PATH = "./data/trafos/label_enc"
FEATS_UNUSED = (
    [
        "Irradiance_gap",
        "Irradiance_dev",
        "TempSta",
        "IrraSta",
        "StaName",
        "StaName_i",
        "TempDiff",
        "IrraDiff",
        "clust",
    ]
    + ["StaName_aq", "Station"]  # Air-quality related features
    + [TID, TARGET]
)
NATURE_CAT_FEATS = ["Location", "Module"]  # Features can't be treated as numeric


class FSArgParser(BaseArgParser):
    """Argument parser for randomly forward feature selection."""

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
            "--pre-selected-feats",
            type=str,
            default=None,
            nargs="*",
            help="manually selected features (forced to use)",
        )
        self.argparser.add_argument(
            "--pre-excluded-feats",
            type=str,
            default=None,
            nargs="*",
            help="manually excluded features (forced to drop)",
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
            "--aux-target",
            type=str,
            default=TARGET,
            help="auxiliary target of the task",
        )


def main(args: Namespace) -> None:
    """Run forward feature selection."""
    model_name = args.model_name
    input_path = args.input_path
    pre_selected_feats = (
        [] if args.pre_selected_feats is None else args.pre_selected_feats
    )
    pre_excluded_feats = (
        [] if args.pre_excluded_feats is None else args.pre_excluded_feats
    )
    cv_scheme = args.cv_scheme
    n_folds = args.n_folds
    oof_size = args.oof_size
    stratified = args.stratified
    aux_target = args.aux_target

    # Prepare data
    data = (
        pd.read_csv(input_path).sort_values(PK).reset_index(drop=True)
    )  # Sort using PK to ensure reproducibility
    with open("./config/dp_template.yaml", "r") as f:
        dp_cfg = yaml.full_load(f)
    feat_base = dp_cfg["feats"]  # Total 48+42 base features are considered before FE
    y = data[aux_target]

    # Drop outliers (top3 extremely large target values)
    ols = []
    ol1 = data[data[TARGET] == 6752].index
    ol2 = data[data[TARGET] == 3765].index
    ol3 = data[data[TARGET] == 3187].index
    for ol in [ol1, ol2, ol3]:
        if len(ol) != 0:
            # Outlier doesn't exist in the current dataset (i.e., specific sampling)
            ols.append(ol[0])
    data = data.drop(ols, axis=0).reset_index(drop=True)

    # Feature engineering
    if model_name != "xgb":
        for enc_file in os.listdir(LABEL_ENC_PATH):
            feat = enc_file.split(".")[0]  # Nature/Pseudo categorical feature
            enc_path = os.path.join(LABEL_ENC_PATH, enc_file)
            with open(enc_path, "rb") as f:  # type: ignore
                enc = pickle.load(f)  # type: ignore
            data[f"{feat}_cat"] = enc.transform(data[feat])

    feat_eng = FE(
        add_month=True,
        add_module_meta=True,
        label_enc=[],  # Taken care by explicit label encoding above
        mine_temp=[
            "TempRange",
            "TempMax2Avg",
            "TempAvg2Min",
            "Temp_m2Temp",
            "TempRangeRatio",
            "TempMax2AvgRatio",
            "TempAvg2MinRatio",
            "Temp_m2TempRatio",
        ],
        mine_irrad=["Irrad_m2Irrad", "Irrad_m2IrradRatio"],
        meta_feats=[],
        knn_meta_feats={},
    )
    data = feat_eng.run(data)

    # Specify categorical features
    if model_name != "xgb":
        cat_feats = ["Month"] + [f for f in data.columns if f.endswith("_cat")]
    else:
        cat_feats = None

    # Build cv iterator
    if cv_scheme == "stratified":
        assert stratified is not None

        enc = LabelEncoder()
        cap_code = enc.fit_transform(data[stratified])
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=168)
        cv_iter = cv.split(data, cap_code, None)
    elif cv_scheme == "gpts":
        cv = GPTSSplit(n_splits=n_folds, n_time_step_oof=oof_size)
        cv_iter = cv.split(data, None, data[TID])

    # Build model
    with open(f"./config/model/{model_name}.yaml", "r") as f:
        model_params = yaml.full_load(f)
    fit_params = model_params["fit_params"]
    model_params = model_params["model_params"]
    if model_name == "lgbm":
        model_params["verbose"] = -1
        model_params["verbose_eval"] = -1
    elif model_name == "xgb":
        model_params["verbosity"] = 0
    elif model_name == "cat":
        model_params["verbose"] = False
    model = build_models(model_name, model_params, 1)[0]

    # Start foward feature selection process
    feats_to_drop = FEATS_UNUSED + NATURE_CAT_FEATS + pre_excluded_feats
    if aux_target is not None:
        feats_to_drop += [aux_target]
    foward_fs = ForwardFS(model, cv_iter, cat_feats)
    foward_fs_wrapper = FSPriorWrapper(
        pre_selected_feats=pre_selected_feats,
        pre_excluded_feats=feats_to_drop,
        fs=foward_fs,
    )
    foward_fs_wrapper.fit(data, data["Generation"], fit_params)  # y, fit_params)

    best_feats = foward_fs_wrapper.fs.best_feats_
    best_prf = foward_fs_wrapper.fs.best_prf_
    print(f"Done.\nFinal selected features: {best_feats}")
    print(f"Best RMSE: {best_prf}")

    # Dump selection results
    if input_path.endswith("season.csv"):
        dump_dir = "season"
    elif input_path.endswith("recency.csv"):
        dump_dir = "recency"
    else:
        dump_dir = "normal"
    dump_path = os.path.join("fs_tmp", dump_dir, f"{gen_exp_id(model_name)}")
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)
    else:
        dump_path = os.path.join("fs_tmp", dump_dir, f"{gen_exp_id(model_name)}")
        os.mkdir(dump_path)

    args = vars(args)
    open(os.path.join(dump_path, f"{best_prf}"), "a").close()
    with open(os.path.join(dump_path, "fs_args.yaml"), "w") as f:
        yaml.dump(args, f)
    with open(os.path.join(dump_path, "feats.txt"), "w") as f:
        for ft in best_feats:
            f.write(f"{ft}\n")


if __name__ == "__main__":
    # Parse arguments
    arg_parser = FSArgParser()
    args = arg_parser.parse()

    # Launch main function
    main(args)
