"""
EDA utilities.
Author: JiaWei Jiang
"""
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import display

colors = sns.color_palette("Set2")


def summarize(
    df: pd.DataFrame,
    file_name: Optional[str] = None,
    n_rows_to_display: Optional[int] = 5,
) -> None:
    """Summarize DataFrame.

    Parameters:
        df: input data
        file_name: name of the input file
        n_rows_to_display: number of rows to display

    Return:
        None
    """
    file_name = "Data" if file_name is None else file_name

    # Derive NaN ratio for each column
    nan_ratio = pd.isna(df).sum() / len(df) * 100
    nan_ratio.sort_values(ascending=False, inplace=True)
    nan_ratio = nan_ratio.to_frame(name="NaN Ratio").T

    # Derive zero ratio for each column
    zero_ratio = (df == 0).sum() / len(df) * 100
    zero_ratio.sort_values(ascending=False, inplace=True)
    zero_ratio = zero_ratio.to_frame(name="Zero Ratio").T

    # Print out summarized information
    print(f"=====Summary of {file_name}=====")
    display(df.head(n_rows_to_display))
    print(f"Shape: {df.shape}")
    print("NaN ratio:")
    display(nan_ratio)
    print("Zero ratio:")
    display(zero_ratio)


def plot_univar_dist(
    data: Union[pd.Series, np.ndarray], feature: str, bins: int = 250
) -> None:
    """Plot univariate distribution.

    Parameters:
        data: univariate data to plot
        feature: feature name of the data
        bins: number of bins
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=data, bins=bins, kde=True, palette=colors, ax=ax)
    ax.axvline(
        x=data.mean(), color="orange", linestyle="dotted", linewidth=1.5, label="Mean"
    )
    ax.axvline(
        x=data.median(),
        color="green",
        linestyle="dotted",
        linewidth=1.5,
        label="Median",
    )
    ax.axvline(
        x=data.mode().values[0],
        color="red",
        linestyle="dotted",
        linewidth=1.5,
        label="Mode",
    )
    ax.set_title(
        f"{feature.upper()} Distibution\n"
        f"Min {round(data.min(), 2)} | "
        f"Max {round(data.max(), 2)} | "
        f"Skewness {round(data.skew(), 2)} | "
        f"Kurtosis {round(data.kurtosis(), 2)}"
    )
    ax.set_xlabel(f"{feature}")
    ax.set_ylabel("Bin Count")
    ax.legend()
    plt.show()


def plot_bivar(
    data: Union[pd.Series, np.ndarray],
    features: Optional[List[str]] = ["0", "1"],
) -> None:
    """Plot bivariate distribution with regression line fitted.

    Parameters:
        data: bivariate data to plot
        features: list of feature names
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    f1, f2 = features[0], features[1]
    corr = data[[f1, f2]].corr().iloc[0, 1]

    ax = sns.jointplot(
        x=data[f1],
        y=data[f2],
        kind="reg",
        height=6,
        marginal_ticks=True,
        joint_kws={"line_kws": {"color": "orange"}},
    )
    ax.fig.suptitle(f"{f1} versus {f2}, Corr={corr:.2}")
    ax.ax_joint.set_xlabel(f1)
    ax.ax_joint.set_ylabel(f2)
    plt.tight_layout()


def plot_gen_map(data: pd.DataFrame) -> None:
    """Plot locations of generators.

    Parameters:
        data: data containing metadata of generators

    Return:
        None
    """
    gen_loc = data.groupby("Location")["Capacity"].unique()
    lat, lon = [], []
    for loc in gen_loc.index:
        lat.append(float(loc.split("-")[0]))
        lon.append(float(loc.split("-")[1]))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lon,
            y=lat,
            mode="markers+text",
            text=gen_loc.values,
            textposition="bottom center",
        )
    )
    fig.update_layout(title="Unique Generator Location")
    fig.show()
