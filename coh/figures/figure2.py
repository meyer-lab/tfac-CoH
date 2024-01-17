"""
This creates Figure 2, tensor factorization of response data.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .common import subplotLabel, getSetup, plot_tFac_CoH, CoH_Scat_Plot, BC_scatter_cells
from ..tensor import factorTensor, BC_status_plot, CoH_LogReg_plot
from ..flow import make_CoH_Tensor, get_status_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    CoH_Data = make_CoH_Tensor(just_signal=True)

    with open('./coh/data/signaling.pkl', 'rb') as ff:
        tFacAllM = pickle.load(ff) # 12 component

    plot_tFac_CoH(ax[0:], tFacAllM, CoH_Data)

    CoH_Data = make_CoH_Tensor(just_signal=True)

    with open('./coh/data/signaling.pkl', 'rb') as ff:
        tFacAllM = pickle.load(ff) # 12 component

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
    ligs = ["IL2-50ng", "IL10-50ng"]
    axes = [7, 9]
    cells = ["CD16 NK", "CD20 B", "CD4+", "CD8+", "CD33 Myeloid", "Treg"]

    for i, sigg in enumerate(["pSTAT5", "pSTAT3"]):
        DF = CoH_DF.loc[(CoH_DF.Marker == sigg) & (CoH_DF.Treatment == ligs[i])]
        DF["Mean"] -= np.nanmean(DF["Mean"].values)
        DF["Mean"] /= np.nanstd(DF["Mean"].values)
        DF = DF[(CoH_DF.Cell.isin(cells))]
        BC_scatter_cells(ax[axes[i]], DF, sigg, ligs[i])
    ax[7].set(ylim=(-3, 4))
    ax[9].set(ylim=(-2, 4))

    # Correlation plot
    comp_corr_plot(tFacAllM, CoH_Data, get_status_df(), ax[8])

    #IL-10 differences

    return f


def cytok_stim_plot(CoH_DF, cytok, cells, ax):
    """Plots cells responses across signaling products for a single stimulatiom"""
    CoH_DF = CoH_DF.loc[(CoH_DF.Treatment == cytok) & (CoH_DF.Cell.isin(cells)) & CoH_DF.Marker.isin(["pSTAT1", "pSTAT3", "pSTAT3", "pSTAT5", "pSTAT6", "pSmad1-2"])]
    sns.boxplot(data=CoH_DF, x="Cell", y="Mean", hue="Marker", palette='husl', showfliers=False, ax=ax)
    ax.set(ylabel="Response to " + cytok)


def comp_corr_plot(tFac, CoH_Array, status_DF, ax):
    """Plots correlation which each component has with outcome across patients"""
    coord = CoH_Array.dims.index("Patient")
    mode_facs = tFac[1][coord]
    Donor_CoH_y = preprocessing.label_binarize(status_DF.Status, classes=['Healthy', 'BC']).flatten()
    corrDF = pd.DataFrame(data=mode_facs, columns=np.arange(1, mode_facs.shape[1] + 1))
    corrDF["BC Status"] = Donor_CoH_y
    corrDF = corrDF.corr()
    corrDF = corrDF.loc["BC Status", :].to_frame()
    corrDF = corrDF.drop("BC Status").reset_index()
    corrDF.columns = ["Component", "BC Correlation"]
    sns.barplot(data=corrDF, y="BC Correlation", x="Component", color='k', ax=ax)
    ax.set(xlim=(-1, 1), ylabel="Component", xlabel="Correlation with BC")
