"""
This creates Figure S4, boxplots of induced responses.
"""
from os.path import join
import pandas as pd
from .common import subplotLabel, getSetup, BC_scatter, BC_scatter_cells, path_here


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    CoH_DF = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    CoH_DF = CoH_DF.loc[CoH_DF.Time == "15min"]
    BC_scatter(ax[0], CoH_DF, "pSTAT3", "IL10-50ng")
    BC_scatter(ax[1], CoH_DF, "pSTAT5", "IL2-50ng")
    BC_scatter(ax[2], CoH_DF, "pSmad1-2", "TGFB-50ng")
    BC_scatter_cells(ax[3], CoH_DF, "pSTAT3", "IL10-50ng", filter=True)
    BC_scatter_cells(ax[4], CoH_DF, "pSTAT5", "IL2-50ng", filter=True)
    BC_scatter_cells(ax[5], CoH_DF, "pSTAT3", "IL6-50ng", filter=True)
    CoH_DF_B = pd.read_csv(join(path_here, "data/CoH_Flow_DF_Basal.csv"))
    CoH_DF_B = CoH_DF_B.loc[CoH_DF_B.Time == "15min"]
    BC_scatter_cells(ax[6], CoH_DF_B, "pSmad1-2", "Untreated", filter=True)
    BC_scatter_cells(ax[7], CoH_DF_B, "pSTAT4", "Untreated", filter=True)
    BC_scatter_cells(ax[8], CoH_DF_B, "pSTAT1", "Untreated", filter=True)

    return f
