"""
This creates Figure S2, factorization of fold-change data.
"""
from ..tensor import (
    factorTensor,
    R2Xplot,
    CoH_LogReg_plot,
    BC_status_plot,
)
from ..flow import make_CoH_Tensor
from .common import subplotLabel, getSetup, BC_scatter_cells_rec
import pandas as pd
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (2, 4))

    # Add subplot labels
    subplotLabel(ax)
    CoH_DF = pd.read_csv("./coh/data/CoH_Rec_DF.csv", index_col=0)

    # Figure A - plot of PD-1 CD8 Cells

    DF = CoH_DF.loc[CoH_DF.Marker == "PD1"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    PD1_DF = DF.loc[DF.Cell.isin(["CD8+", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 Naive", "CD8 TEMRA"])]
    BC_scatter_cells_rec(ax[0], PD1_DF, "PD1", filter=False)
    

    # B PD-L1 CD8 and B cells

    DF = CoH_DF.loc[CoH_DF.Marker == "PD_L1"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    PDL1_DF = DF.loc[DF.Cell.isin(["CD8+", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 Naive", "CD8 TEMRA", "CD20 B", "CD20 B Naive", "CD20 B Memory"])]
    BC_scatter_cells_rec(ax[1], PDL1_DF, "PD_L1", filter=False)

    # C IL6Ra B

    DF = CoH_DF.loc[CoH_DF.Marker == "IL6Ra"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    IL6Ra_DF = DF.loc[DF.Cell.isin(["CD8+", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 Naive", "CD8 TEMRA", "CD20 B", "CD20 B Naive", "CD20 B Memory"])]
    BC_scatter_cells_rec(ax[2], IL6Ra_DF, "IL6Ra", filter=False)

    # D IL2Ra Tregs

    DF = CoH_DF.loc[CoH_DF.Marker == "IL2Ra"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    IL2Ra_DF = DF.loc[DF.Cell.isin(["Treg"])]
    BC_scatter_cells_rec(ax[3], IL2Ra_DF, "IL2Ra", filter=False)

    # E PD-L1 in B vs CD8 Cells

    # F IL6Ra in B vs CD8 Cells

    # G IL2Ra Tregs vs PD-L1 CD8s

    # H IL2Ra Tregs vs IL6Ra B

    # I Univariate vs coordinated ROC

    return f
