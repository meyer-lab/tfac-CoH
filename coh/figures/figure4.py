"""
This creates Figure 1.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from tensorpack.cmtf import cp_normalize
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..tensor import factorTensor, CoH_LogReg_plot, plot_tFac_CoH, make_alldata_DF, BC_status_plot, BC_scatter
from ..flow import make_flow_df, make_CoH_Tensor
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df(foldChange=True)
    #make_CoH_Tensor(just_signal=True, foldChange=True)
    num_comps = 12

    # make_alldata_DF(CoH_Data, PCA=False, foldChange=True)
    CoH_DF = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    BC_scatter(ax[0], CoH_DF, "pSTAT3", "IL10-50ng")
    BC_scatter(ax[1], CoH_DF, "pSTAT5", "IL2-50ng", cells=["Treg", "CD4+", "CD8+"])

    return f
