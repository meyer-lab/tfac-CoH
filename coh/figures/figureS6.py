"""
This creates Figure S6.
"""
import pandas as pd
from .figureCommon import subplotLabel, getSetup, BC_scatter, BC_scatter_cells
from os.path import join, dirname

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 12), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df(foldChange=True)
    #make_CoH_Tensor(just_signal=True, foldChange=True)

    # make_alldata_DF(CoH_Data, PCA=False, foldChange=True)
    CoH_DF = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    BC_scatter(ax[0], CoH_DF, "pSTAT3", "IL10-50ng")
    BC_scatter(ax[1], CoH_DF, "pSTAT5", "IL2-50ng")
    BC_scatter_cells(ax[2], CoH_DF, "pSTAT3", "IL10-50ng", filter=False)
    BC_scatter_cells(ax[3], CoH_DF, "pSTAT5", "IL2-50ng", filter=False)
    CoH_DF_B = pd.read_csv(join(path_here, "data/CoH_Flow_DF_Basal.csv"))
    BC_scatter_cells(ax[4], CoH_DF_B, "pSmad1-2", "Untreated", filter=False)
    BC_scatter_cells(ax[5], CoH_DF_B, "pSTAT4", "Untreated", filter=False)

    return f
