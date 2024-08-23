"""
This creates Figure S5, boxplots of induced responses.
"""

import pandas as pd
import numpy as np
from .common import subplotLabel, getSetup, BC_scatter_cells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 12), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    CoH_DF = pd.read_csv("./coh/data/CoH_Flow_DF.csv")
    CoH_DF = CoH_DF.loc[CoH_DF.Time == "15min"]
    ligs = ["IFNg-50ng", "IL10-50ng", "IL6-50ng", "IL2-50ng", "IL4-50ng", "TGFB-50ng"]

    for i, sigg in enumerate(
        ["pSTAT1", "pSTAT3", "pSTAT3", "pSTAT5", "pSTAT6", "pSmad1-2"]
    ):
        DF = CoH_DF.loc[(CoH_DF.Marker == sigg) & (CoH_DF.Treatment == ligs[i])]
        DF["Mean"] -= np.nanmean(DF["Mean"].values)
        DF["Mean"] /= np.nanstd(DF["Mean"].values)
        BC_scatter_cells(ax[i], DF, sigg, ligs[i])

    CoH_DF_B = pd.read_csv("./coh/data/CoH_Flow_DF_Basal.csv")
    CoH_DF_B = CoH_DF_B.loc[CoH_DF_B.Time == "15min"]

    for i, sigg in enumerate(
        ["pSTAT1", "pSTAT3", "pSTAT4", "pSTAT5", "pSTAT6", "pSmad1-2"]
    ):
        DF = CoH_DF_B.loc[
            (CoH_DF_B.Marker == sigg) & (CoH_DF_B.Treatment == "Untreated")
        ]
        DF["Mean"] -= np.nanmean(DF["Mean"].values)
        DF["Mean"] /= np.nanstd(DF["Mean"].values)
        BC_scatter_cells(ax[6 + i], DF, sigg, "Untreated")

    return f
