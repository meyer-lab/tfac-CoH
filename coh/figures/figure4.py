"""
This creates Figure 4.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from tensorpack.cmtf import cp_normalize
from .figureCommon import subplotLabel, getSetup, BC_scatter, BC_scatter_cells
from os.path import join, dirname
from ..flow import make_flow_df, make_CoH_Tensor
from ..tensor import get_status_dict
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 16), (4, 2))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df(foldChange=True)
    #make_CoH_Tensor(just_signal=True, foldChange=True)
    # make_alldata_DF(CoH_Data, PCA=False, foldChange=True)

    CoH_DF = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    BC_scatter(ax[0], CoH_DF, "pSTAT3", "IL10-50ng")
    BC_scatter(ax[1], CoH_DF, "pSTAT5", "IL2-50ng")
    BC_scatter_cells(ax[2], CoH_DF, "pSTAT3", "IL10-50ng", filter=True)
    BC_scatter_cells(ax[3], CoH_DF, "pSTAT5", "IL2-50ng", filter=True)
    CoH_DF_B = pd.read_csv(join(path_here, "data/CoH_Flow_DF_Basal.csv"))
    BC_scatter_cells(ax[4], CoH_DF_B, "pSmad1-2", "Untreated", filter=True)
    BC_scatter_cells(ax[5], CoH_DF_B, "pSTAT4", "Untreated", filter=True)
    dysreg_cor_plot(ax[6], CoH_DF, "IL10-50ng", "pSTAT3", "IL2-50ng", "pSTAT5")
    dysreg_cor_plot(ax[7], CoH_DF, "IL10-50ng", "pSTAT3", "Untreated", "pSmad1-2", CoH_DF_B)

    return f


def dysreg_cor_plot(ax, CoH_DF, cytokine1, marker1, cytokine2, marker2, CoH_DF_B=False):
    """Plots possible correlation of dysregulation"""
    status_dict = get_status_dict()
    CoH_DF1 = CoH_DF.loc[(CoH_DF.Treatment == cytokine1) & (CoH_DF.Marker == marker1)]
    if type(CoH_DF_B) != pd.DataFrame:
        CoH_DF2 = CoH_DF.loc[(CoH_DF.Treatment == cytokine2) & (CoH_DF.Marker == marker2)]
    else:
        CoH_DF2 = CoH_DF_B.loc[(CoH_DF_B.Treatment == cytokine2) & (CoH_DF_B.Marker == marker2)]
    CoH_DF1 = CoH_DF1.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()
    CoH_DF1["Status"] = CoH_DF1.replace({"Patient": status_dict}).Patient.values
    CoH_DF1 = CoH_DF1.drop(["Marker", "Cell"], axis=1).rename({"Mean": marker1}, axis=1)

    CoH_DF2 = CoH_DF2.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()
    CoH_DF2["Status"] = CoH_DF2.replace({"Patient": status_dict}).Patient.values
    CoH_DF2 = CoH_DF2.drop(["Marker", "Cell"], axis=1).rename({"Mean": marker2}, axis=1)
    
    CoH_DF2[marker1] = CoH_DF1[marker1].values
    sns.scatterplot(data=CoH_DF2, x=marker1, y=marker2, hue="Status", ax=ax)
