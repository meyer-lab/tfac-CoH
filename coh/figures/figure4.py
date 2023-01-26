"""
This creates Figure 4.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from tensorpack.cmtf import cp_normalize
from .figureCommon import subplotLabel, getSetup, BC_scatter, BC_scatter_cells, BC_scatter_ligs
from os.path import join, dirname
from ..flow import make_flow_df, make_CoH_Tensor
from ..tensor import get_status_dict
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (4, 3))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df(foldChange=True)
    # make_CoH_Tensor(just_signal=True, foldChange=True)

    #make_alldata_DF(CoH_Data, PCA=False, foldChange=True)
    CoH_DF = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    BC_scatter(ax[0], CoH_DF, "pSTAT3", "IL10-50ng")
    BC_scatter(ax[1], CoH_DF, "pSTAT5", "IL2-50ng")
    BC_scatter_cells(ax[2], CoH_DF, "pSTAT3", filter=True)
    BC_scatter_cells(ax[3], CoH_DF, "pSTAT5", filter=True)
    CoH_DF_B = pd.read_csv(join(path_here, "data/CoH_Flow_DF_Basal.csv"))
    BC_scatter_cells(ax[4], CoH_DF_B, "pSmad1-2", "Untreated", filter=True)
    BC_scatter_cells(ax[5], CoH_DF_B, "pSTAT4", "Untreated", filter=True)
    dysreg_cor_plot(ax[6], CoH_DF, "IL10-50ng", "pSTAT3", "IL2-50ng", "pSTAT5")
    dysreg_cor_plot(ax[7], CoH_DF, "IL10-50ng", "pSTAT3", "Untreated", "pSmad1-2", CoH_DF_B)
    dysreg_cor_plot(ax[8], CoH_DF, "IL10-50ng", "pSTAT3", "Untreated", "pSTAT4", CoH_DF_B)
    dysreg_cor_plot(ax[9], CoH_DF, "IL2-50ng", "pSTAT5", "Untreated", "pSmad1-2", CoH_DF_B)
    dysreg_cor_plot(ax[10], CoH_DF, "IL2-50ng", "pSTAT5", "Untreated", "pSTAT4", CoH_DF_B)
    dysreg_cor_plot(ax[11], CoH_DF_B, "Untreated", "pSmad1-2", "Untreated", "pSTAT4", CoH_DF_B)

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
    Healthy_DF = CoH_DF2.loc[CoH_DF2.Status == "BC"]
    BC_DF = CoH_DF2.loc[CoH_DF2.Status == "Healthy"]
    print(marker1, marker2)
    print(spearmanr(CoH_DF2[marker1], CoH_DF2[marker2]), " Overall")
    print(spearmanr(Healthy_DF[marker1], Healthy_DF[marker2]), " Healthy")
    print(spearmanr(BC_DF[marker1], BC_DF[marker2]), " BC")

    sns.scatterplot(data=CoH_DF2, x=marker1, y=marker2, hue="Status", ax=ax)
    #ax.text(5, np.amax(CoH_DF2[marker2].values) * 1.1, str(spearmanr(CoH_DF2[marker1], CoH_DF2[marker2])[0]) + " Overall Spearman")
    #ax.text(5, np.amax(CoH_DF2[marker2].values) * 1, str(spearmanr(Healthy_DF[marker1], Healthy_DF[marker2])[0]) + " Healthy Spearman")
    #ax.text(5, np.amax(CoH_DF2[marker2].values) * 0.9, str(spearmanr(BC_DF[marker1], BC_DF[marker2])[0]) + " BC Spearman")
    ax.set(xlabel=marker1 + " response to " + cytokine1, ylabel=marker2 + " response to " + cytokine2)


def resp_bar(ax, CoH_DF, cells, marker):
    """Use for seeing which patients are diverging, not actual figure"""
    CoH_DF = CoH_DF.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()
    CoH_DF = CoH_DF.loc[(CoH_DF.Cell.isin(cells)) & (CoH_DF.Marker == marker)]
    sns.barplot(data=CoH_DF, x="Patient", y="Mean", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
