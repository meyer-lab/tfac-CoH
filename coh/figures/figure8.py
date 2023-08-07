"""
This creates Figure 4.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as plt
from scipy.stats import zscore
from os.path import join, dirname
from ..flow import make_CoH_Tensor, get_status_df
from .common import subplotLabel, getSetup, BC_scatter_cells_rec


plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    CoH_DF = pd.read_csv("./coh/data/CoH_Flow_DF.csv", index_col=0)
    CoH_DF = CoH_DF.loc[CoH_DF.Patient != "Patient 406"]
    treatments = np.array(["IL2-50ng", "IL4-50ng", "IL6-50ng", "IL10-50ng", "IFNg-50ng", "TGFB-50ng", "IFNg-50ng+IL6-50ng", "Untreated"])
    markers = np.array(["pSTAT1", "pSTAT3", "pSTAT4", "pSTAT5", "pSTAT6", "pSmad1-2"])
    CoH_DF = CoH_DF.loc[(CoH_DF.Treatment.isin(treatments)) & (CoH_DF.Marker.isin(markers)) & (CoH_DF.Time == "15min") & (CoH_DF.Treatment != "Untreated")].dropna()
    
    CoH_DF = CoH_DF.groupby(["Patient", "Cell", "Treatment", "Marker"]).mean().reset_index()
    CoH_DF = (CoH_DF.pivot(index=["Patient", "Cell", "Treatment"], columns="Marker", values="Mean").reset_index().set_index("Patient"))
    CoH_DF.iloc[:, 2::] = CoH_DF.iloc[:, 2::].apply(zscore)

    CoH_DF_R = pd.read_csv("./coh/data/CoH_Rec_DF.csv", index_col=0).dropna()
    CoH_DF_R = CoH_DF_R.loc[CoH_DF_R.Patient != "Patient 19186-12"]
    CoH_DF_R = CoH_DF_R.groupby(["Patient", "Cell", "Marker"]).mean().reset_index()
    CoH_DF_R = (
        CoH_DF_R.pivot(index=["Patient", "Cell"], columns="Marker", values="Mean")
        .reset_index()
        .set_index("Patient")
    )
    CoH_DF_R.loc[:, CoH_DF_R.columns.values != "Cell"] = CoH_DF_R.loc[
        :, CoH_DF_R.columns.values != "Cell"
    ].apply(zscore)

    # A B - response vs receptor across cell types
    plot_rec_resp_cell(CoH_DF, CoH_DF_R, "IFNg R1", "pSTAT1", "IFNg-50ng", ax[0])
    plot_rec_resp_cell(CoH_DF, CoH_DF_R, "IL2Ra", "pSTAT5", "IL2-50ng", ax[1])

    # C IL-10R by BC status
    recDF = pd.read_csv("./coh/data/CoH_Rec_DF.csv", index_col=0)
    DF = recDF.loc[recDF.Marker == "IL10R"]
    #DF["Mean"] -= np.mean(DF["Mean"].values)
    #DF["Mean"] /= np.std(DF["Mean"].values)
    BC_scatter_cells_rec(ax[2], DF, "IL10R", filter=False)

    # D Response to IL10 per patient in monocytes

    rec_vs_induced(CoH_DF, CoH_DF_R, receptor= "IL10R", marker="pSTAT3", treatment="IL10-50ng", cell="Classical Monocyte", ax=ax[3])
    
    # E-F IL2Ra and IL2RB by BC status

    DF = recDF.loc[recDF.Marker == "IL2Ra"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    BC_scatter_cells_rec(ax[4], DF, "IL2Ra", filter=False)

    DF = recDF.loc[recDF.Marker == "IL2RB"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    BC_scatter_cells_rec(ax[5], DF, "IL2RB", filter=False)

    rec_vs_induced(CoH_DF, CoH_DF_R, receptor= "IL2Ra", marker="pSTAT5", treatment="IL2-50ng", cell="CD8+", ax=ax[6])
    rec_vs_induced(CoH_DF, CoH_DF_R, receptor= "IL2Ra", marker="pSTAT5", treatment="IL2-50ng", cell="Treg", ax=ax[7])
    rec_vs_induced(CoH_DF, CoH_DF_R, receptor= "IL2RB", marker="pSTAT5", treatment="IL2-50ng", cell="CD8+", ax=ax[8])
    rec_vs_induced(CoH_DF, CoH_DF_R, receptor= "IL2RB", marker="pSTAT5", treatment="IL10-50ng", cell="Treg", ax=ax[9])


   

    return f


def rec_vs_induced(CoH_DF, CoH_DF_R, receptor, marker, treatment, cell, ax):
    """Plots receptor level vs response to treatment"""
    sigDF = CoH_DF.loc[(CoH_DF.Treatment == treatment) & (CoH_DF.Cell == cell)][marker].to_frame()
    recDF = CoH_DF_R.loc[(CoH_DF_R.Cell == cell)][receptor].to_frame()
    jointDF = sigDF.join(recDF)

    status_DF = get_status_df().set_index("Patient")
    jointDF = jointDF.join(status_DF)
    sns.scatterplot(data=jointDF, x=receptor, y=marker, hue="Status", ax=ax)
    ax.set(xlabel=receptor + " in " + cell, ylabel = marker + " in " + treatment + " in " + cell)


def plot_rec_resp_cell(sigDF, recDF, receptor, marker, treatment, ax):
    """Plots receptor in pop 1 vs receptor in pop 2 per patient, by disease status"""
    status_DF = get_status_df()
    sigDF = sigDF.reset_index().set_index("Patient").join(status_DF.set_index("Patient"), on="Patient")
    plotDF = pd.DataFrame()
    for cell in sigDF.Cell.unique():
        rec = np.mean(recDF.loc[(recDF.Cell == cell)][receptor].values)
        resp = np.mean(sigDF.loc[(sigDF.Cell == cell) & (sigDF.Treatment == treatment)][marker].values)
        plotDF = pd.concat([plotDF, pd.DataFrame({"Cell": [cell], receptor: rec, marker + " response to " + treatment: resp})])

    sns.scatterplot(data=plotDF, x=receptor, y=marker + " response to " + treatment, hue="Cell", ax=ax)