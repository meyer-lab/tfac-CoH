"""This creates Figure 2, tensor factorization of response data."""

import pickle

import numpy as np
import pandas as pd
import seaborn as sns

from ..flow import get_status_df, make_CoH_Tensor
from ..tensor import BC_status_plot, CoH_LogReg_plot
from .common import (
    BC_scatter_cells,
    comp_corr_plot,
    getSetup,
    plot_tFac_CoH,
    subplotLabel,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    CoH_Data = make_CoH_Tensor(just_signal=True)

    with open("./coh/data/signaling.pkl", "rb") as ff:
        tFacAllM = pickle.load(ff)  # 12 component

    plot_tFac_CoH(ax[0:], tFacAllM, CoH_Data)

    CoH_Data = make_CoH_Tensor(just_signal=True)

    with open("./coh/data/signaling.pkl", "rb") as ff:
        tFacAllM = pickle.load(ff)  # 12 component

    BC_status_plot(13, CoH_Data, ax[5], get_status_df())
    CoH_LogReg_plot(ax[6], tFacAllM, CoH_Data, get_status_df())

    # B cells and Tregs in response to different stimulations
    CoH_DF = pd.read_csv("./coh/data/CoH_Flow_DF.csv")
    CoH_DF = CoH_DF.loc[CoH_DF.Time == "15min"]
    cytok_stim_plot(CoH_DF, "IL4-50ng", ["CD20 B", "Treg"], ax=ax[6])
    ax[6].set(ylim=(-750, 1500))

    # IFNg by status
    CoH_DF = pd.read_csv("./coh/data/CoH_Flow_DF.csv")
    CoH_DF = CoH_DF.loc[CoH_DF.Time == "15min"]
    ligs = ["IL2-50ng", "IL10-50ng", "Untreated"]
    axes = [7, 9, 10]
    cells = ["CD16 NK", "CD20 B", "CD4+", "CD8+", "CD33 Myeloid", "Treg"]

    for i, sigg in enumerate(["pSTAT5", "pSTAT3", "pSTAT4"]):
        DF = CoH_DF.loc[(CoH_DF.Marker == sigg) & (CoH_DF.Treatment == ligs[i])]
        DF["Mean"] -= np.nanmean(DF["Mean"].values)
        DF["Mean"] /= np.nanstd(DF["Mean"].values)
        DF = DF[(CoH_DF.Cell.isin(cells))]
        BC_scatter_cells(ax[axes[i]], DF, sigg, ligs[i])
    ax[7].set(ylim=(-3, 4))
    ax[9].set(ylim=(-2, 4))
    ax[10].set(ylim=(-4, 3))

    # Correlation plot
    comp_corr_plot(tFacAllM, CoH_Data, get_status_df(), ax[8])

    # IL-10 differences

    return f


def cytok_stim_plot(CoH_DF, cytok, cells, ax) -> None:
    """Plots cells responses across signaling products for a single stimulatiom."""
    CoH_DF = CoH_DF.loc[
        (CoH_DF.Treatment == cytok)
        & (CoH_DF.Cell.isin(cells))
        & CoH_DF.Marker.isin(
            ["pSTAT1", "pSTAT3", "pSTAT3", "pSTAT5", "pSTAT6", "pSmad1-2"],
        )
    ]
    sns.boxplot(
        data=CoH_DF,
        x="Cell",
        y="Mean",
        hue="Marker",
        palette="husl",
        showfliers=False,
        ax=ax,
    )
    ax.set(ylabel="Response to " + cytok)
