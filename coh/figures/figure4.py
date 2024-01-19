"""
This creates Figure 4, tensor factorization of receptor data.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup, plot_tFac_CoH, BC_scatter_cells_rec
from ..tensor import factorTensor, CoH_LogReg_plot, BC_status_plot
from ..flow_rec import make_CoH_Tensor_rec, get_status_rec_df



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((7, 7), (3, 3))

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    CoH_Data = make_CoH_Tensor_rec()
    tFacAllM = factorTensor(CoH_Data.to_numpy(), r=5)
    
    plot_tFac_CoH(ax[0:], tFacAllM, CoH_Data)

    CoH_Data_R = make_CoH_Tensor_rec()
    tFacAllM_R = factorTensor(CoH_Data_R.to_numpy(), r=5)

    BC_status_plot(6, CoH_Data_R, ax[3], get_status_rec_df())
    CoH_LogReg_plot(ax[4], tFacAllM_R, CoH_Data_R, get_status_rec_df())
    

    CoH_Data_DF = pd.read_csv("./coh/data/CoH_Rec_DF.csv")

    DF = CoH_Data_DF.loc[CoH_Data_DF.Marker == "IL7Ra"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    rec_cell_plot(DF, "IL7Ra", ["CD16 NK", "CD20 B", "CD4+", "CD8+", "CD33 Myeloid", "Treg"], ax[5])
    ax[5].set(ylim=(-4, 2))

    DF = CoH_Data_DF.loc[CoH_Data_DF.Marker == "PD1"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    PD1_DF = DF.loc[
        DF.Cell.isin(
            ["CD8+", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 Naive", "CD8 TEMRA"]
        )
    ]
    BC_scatter_cells_rec(ax[6], PD1_DF, "PD1", filter=False)

    DF = CoH_Data_DF.loc[CoH_Data_DF.Marker == "PD_L1"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    PDL1_DF = DF.loc[
        DF.Cell.isin(
            [
                "CD8+",
                "CD8 TEM",
                "CD8 TCM",
                "CD8 Naive",
                "CD8 Naive",
                "CD8 TEMRA",
                "CD20 B",
                "CD20 B Naive",
                "CD20 B Memory",
            ]
        )
    ]
    BC_scatter_cells_rec(ax[7], PDL1_DF, "PD_L1", filter=False)


    DF = CoH_Data_DF.loc[CoH_Data_DF.Marker == "IL6Ra"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    IL6Ra_DF = DF.loc[
        DF.Cell.isin(
            [
                "CD8+",
                "CD8 TEM",
                "CD8 TCM",
                "CD8 Naive",
                "CD8 Naive",
                "CD8 TEMRA",
                "CD20 B",
                "CD20 B Naive",
                "CD20 B Memory",
            ]
        )
    ]
    BC_scatter_cells_rec(ax[8], IL6Ra_DF, "IL6Ra", filter=False)

    return f


def rec_cell_plot(CoH_DF, rec, cells, ax):
    """Plots cells responses across signaling products for a single stimulatiom"""
    CoH_DF = CoH_DF.loc[(CoH_DF.Marker == rec) & (CoH_DF.Cell.isin(cells))]
    sns.boxplot(data=CoH_DF, x="Cell", y="Mean", palette='husl', showfliers=False, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set(ylabel=rec)
