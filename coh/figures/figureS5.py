"""
This creates Figure S5, boxplots of induced responses.
"""
import pandas as pd
from .common import subplotLabel, getSetup, BC_scatter_cells


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 12), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    CoH_DF = pd.read_csv("./coh/data/CoH_Flow_DF.csv")
    CoH_DF = CoH_DF.loc[CoH_DF.Time == "15min"]

    BC_scatter_cells(ax[0], CoH_DF, "pSTAT1", "IFNg-50ng")
    BC_scatter_cells(ax[1], CoH_DF, "pSTAT3", "IL10-50ng")
    BC_scatter_cells(ax[2], CoH_DF, "pSTAT4", "IFNg-50ng")
    BC_scatter_cells(ax[3], CoH_DF, "pSTAT5", "IL2-50ng")
    BC_scatter_cells(ax[4], CoH_DF, "pSTAT6", "IL4-50ng")
    BC_scatter_cells(ax[5], CoH_DF, "pSmad1-2", "TGFB-50ng")

    CoH_DF_B = pd.read_csv("./coh/data/CoH_Flow_DF_Basal.csv")
    CoH_DF_B = CoH_DF_B.loc[CoH_DF_B.Time == "15min"]

    for i, sigg in enumerate(["pSTAT1", "pSTAT3", "pSTAT4", "pSTAT5", "pSTAT6", "pSmad1-2"]):
        BC_scatter_cells(ax[6 + i], CoH_DF_B, sigg, "Untreated")

    return f
