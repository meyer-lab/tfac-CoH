"""
This creates Figure 5.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from .figureCommon import subplotLabel, getSetup, BC_scatter_cells_rec
from os.path import join, dirname
from ..flow_rec import make_flow_df_rec, make_CoH_Tensor_rec
from ..tensor import CoH_LogReg_plot, BC_status_plot_rec, get_status_dict_rec, make_allrec_DF

plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), (2, 3))

    # Add subplot labels
    subplotLabel(ax)
    #make_flow_df_rec()
    #make_CoH_Tensor_rec()

    #make_allrec_DF(CoH_Data_R)
    #matrix_DF_R = pd.read_csv(join(path_here, "data/CoH_Matrix_Rec.csv"), index_col=0).dropna(axis='columns').set_index("Patient")
    #BC_status_plot_rec(20, CoH_Data_R, matrix_DF_R, ax[1])
    CoH_Data_DF_R = pd.read_csv(join(path_here, "data/CoH_Rec_DF.csv"))
    filt_list = [False, True, True, True, True, True]

    for i, rec in enumerate(np.array(["IL10R", "IL2RB", "IL12RI", "TGFB RII", "PD_L1", "IL6Ra"])):
        BC_scatter_cells_rec(ax[i], CoH_Data_DF_R, rec, filter=filt_list[i])

    """
    CoH_DF = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    CoH_DF_B = pd.read_csv(join(path_here, "data/CoH_Flow_DF_Basal.csv"))
    dysreg_cor_plot_rec(ax[4], CoH_Data_DF_R, "IL6Ra", "IL10R", "N/A")
    dysreg_cor_plot_rec(ax[5], CoH_Data_DF_R, "IL6Ra", "PD_L1", "N/A")
    dysreg_cor_plot_rec(ax[6], CoH_Data_DF_R, "IL10R", "pSTAT3", "IL10-50ng", CoH_DF, cells=["CD8 Naive", "CD8 TCM", "CD8 TEM", "CD8 TEMRA"])
    dysreg_cor_plot_rec(ax[7], CoH_Data_DF_R, "IL12RI", "pSTAT4", "Untreated", CoH_DF_B)
    """

    return f


def dysreg_cor_plot_rec(ax, CoH_DF_rec, marker1, marker2, cytokine2, CoH_DF=False, cells=False):
    """Plots possible correlation of dysregulation"""
    status_dict = get_status_dict_rec()
    CoH_DF1 = CoH_DF_rec.loc[(CoH_DF_rec.Marker == marker1)]
    if type(CoH_DF) != pd.DataFrame:
        CoH_DF2 = CoH_DF_rec.loc[(CoH_DF_rec.Marker == marker2)]
    else:
        CoH_DF = CoH_DF.loc[CoH_DF.Patient != "Patient 406"]
        CoH_DF1 = CoH_DF1.loc[CoH_DF1.Patient != "Patient 19186-12"]
        CoH_DF2 = CoH_DF.loc[(CoH_DF.Treatment == cytokine2) & (CoH_DF.Marker == marker2)]
    CoH_DF1 = CoH_DF1.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()
    CoH_DF1["Status"] = CoH_DF1.replace({"Patient": status_dict}).Patient.values
    if type(cells) == list:
        CoH_DF1 = CoH_DF1.loc[CoH_DF1.Cell.isin(cells)]
    CoH_DF1 = CoH_DF1.drop(["Marker", "Cell"], axis=1).rename({"Mean": marker1}, axis=1)
    

    CoH_DF2 = CoH_DF2.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()
    CoH_DF2["Status"] = CoH_DF2.replace({"Patient": status_dict}).Patient.values
    if type(cells) == list:
        CoH_DF2 = CoH_DF2.loc[CoH_DF2.Cell.isin(cells)]
    CoH_DF2 = CoH_DF2.drop(["Marker", "Cell"], axis=1).rename({"Mean": marker2}, axis=1)
    
    CoH_DF2[marker1] = CoH_DF1[marker1].values
    Healthy_DF = CoH_DF2.loc[CoH_DF2.Status == "BC"]
    BC_DF = CoH_DF2.loc[CoH_DF2.Status == "Healthy"]
    print(marker1, marker2)
    print(spearmanr(CoH_DF2[marker1], CoH_DF2[marker2]), " Overall")
    print(spearmanr(Healthy_DF[marker1], Healthy_DF[marker2]), " Healthy")
    print(spearmanr(BC_DF[marker1], BC_DF[marker2]), " BC")

    sns.scatterplot(data=CoH_DF2, x=marker1, y=marker2, hue="Status", ax=ax)
    if type(CoH_DF) != pd.DataFrame:
        ax.set(xlabel=marker1, ylabel=marker2)
    else:
        ax.set(xlabel=marker1, ylabel=marker2 + " response to " + cytokine2)


def rec_bar(ax, CoH_DF_R, cells, marker):
    """Use for seeing which patients are diverging, not actual figure"""
    CoH_DF_R = CoH_DF_R.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()
    CoH_DF_R = CoH_DF_R.loc[(CoH_DF_R.Cell.isin(cells)) & (CoH_DF_R.Marker == marker)]
    sns.barplot(data=CoH_DF_R, x="Patient", y="Mean", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
