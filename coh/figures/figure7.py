"""
This creates Figure 3, dissection of signaling.
"""
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from .common import subplotLabel, getSetup, BC_scatter_cells
from ..flow import make_CoH_Tensor, get_status_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (3, 4))

    # Add subplot labels
    subplotLabel(ax)
    CoH_DF = pd.read_csv("./coh/data/CoH_Flow_DF.csv", index_col=0)
    treatments = np.array(["IL2-50ng", "IL4-50ng", "IL6-50ng", "IL10-50ng", "IFNg-50ng", "TGFB-50ng", "IFNg-50ng+IL6-50ng", "Untreated"])
    CoH_DF = CoH_DF.loc[CoH_DF.Treatment.isin(treatments)].dropna()

    # Figure A Markers for signaling component

    CoH_Data = make_CoH_Tensor(just_signal=True)

    with open('./coh/data/signaling.pkl', 'rb') as ff:
        tFacAllM = pickle.load(ff) # 12 component
    mode = CoH_Data.dims[1]
    tFacDF = pd.DataFrame(tFacAllM.factors[1], index=CoH_Data.coords[mode], columns=[str(i + 1) for i in range(tFacAllM.factors[1].shape[1])])
    cmap = sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(data=tFacDF, ax=ax[0], cmap=cmap, vmin=-1, vmax=1, cbar=(True))

    # B Baseline pSTAT3

    BC_scatter_cells(ax[1], CoH_DF, "pSTAT3", "Untreated")

    # C Baseline pSTAT3 vs pSTAT3 induced

    meanDF = CoH_DF.groupby(["Patient", "Cell", "Treatment", "Marker"]).mean().reset_index()

    meanDF = (meanDF.pivot(index=["Patient", "Cell", "Treatment"], columns="Marker", values="Mean").reset_index().set_index("Patient"))
    meanDF.iloc[:, 2::] = meanDF.iloc[:, 2::].apply(zscore)

    plot_by_patient(
        meanDF,
        cell1="CD8+",
        receptor1="pSTAT3",
        treatment1="Untreated",
        cell2="CD8+",
        receptor2="pSTAT3",
        treatment2="IL10-50ng",
        ax=ax[2],
    )

    # D CD8+ pSTAT3 vs B pSTAT3 in IL10

    plot_by_patient(
        meanDF,
        cell1="CD8 TCM",
        receptor1="pSTAT3",
        treatment1="IL10-50ng",
        cell2="CD20 B",
        receptor2="pSTAT3",
        treatment2="IL10-50ng",
        ax=ax[3],
    )
    
    # E CD8+ pSTAT3 vs B pSTAT3 in IL10

    plot_by_patient(
        meanDF,
        cell1="CD8+",
        receptor1="pSTAT5",
        treatment1="Untreated",
        cell2="CD8+",
        receptor2="pSTAT5",
        treatment2="IL2-50ng",
        ax=ax[4],
    )
    

    # F CD8+ pSTAT5 vs Treg pSTAT5 in IL2

    plot_by_patient(
        meanDF,
        cell1="CD8+",
        receptor1="pSTAT5",
        treatment1="IL2-50ng",
        cell2="Treg",
        receptor2="pSTAT5",
        treatment2="IL2-50ng",
        ax=ax[5],
    )

    # G CD8+ pSTAT3 vs B pSTAT3 in IL10

    plot_by_patient(
        meanDF,
        cell1="CD8+",
        receptor1="pSTAT5",
        treatment1="IL2-50ng",
        cell2="Treg",
        receptor2="pSTAT5",
        treatment2="IL2-50ng",
        ax=ax[6],
    )

    # H Untreated pSTAT4 and pSMAD1-2 by cell

    meanDF_cell = CoH_DF.groupby(["Cell", "Treatment", "Marker"]).mean().reset_index()

    meanDF_cell = (meanDF_cell.pivot(index=["Cell", "Treatment"], columns="Marker", values="Mean").reset_index().set_index("Cell"))
    meanDF_cell.iloc[:, 2::] = meanDF_cell.iloc[:, 2::].apply(zscore)

    plot_by_cell(
        meanDF_cell,
        receptor1="pSTAT4",
        treatment1="Untreated",
        receptor2="pSmad1-2",
        treatment2="Untreated",
        ax=ax[7],
    )

    # I pSTAT4 vs pSMAD untreated per patient

    meanDF_pat = CoH_DF.groupby(["Patient", "Treatment", "Marker"]).mean().reset_index()

    meanDF_pat = (meanDF_pat.pivot(index=["Patient", "Treatment"], columns="Marker", values="Mean").reset_index().set_index("Patient"))
    meanDF_pat.iloc[:, 2::] = meanDF_pat.iloc[:, 2::].apply(zscore)

    plot_per_patient(
        meanDF_pat,
        receptor1="pSTAT4",
        treatment1="Untreated",
        receptor2="pSmad1-2",
        treatment2="Untreated",
        ax=ax[8],
    )

    return f


def plot_by_patient(sigDF, cell1, receptor1, treatment1, cell2, receptor2, treatment2, ax):
    """Plots receptor in pop 1 vs receptor in pop 2 per patient, by disease status"""
    status_DF = get_status_df()
    plotDF = pd.DataFrame({"Patient": sigDF.loc[(sigDF.Cell == cell1) & (sigDF.Treatment == treatment1)].index.values})
    plotDF[cell1 + " " + receptor1 + " " + treatment1] = sigDF.loc[(sigDF.Cell == cell1) & (sigDF.Treatment == treatment1)][receptor1].values
    plotDF[cell2 + " " + receptor2 + " " + treatment2] = sigDF.loc[(sigDF.Cell == cell2) & (sigDF.Treatment == treatment2)][receptor2].values
    plotDF = plotDF.set_index("Patient").join(
        status_DF.set_index("Patient"), on="Patient"
    )
    sns.scatterplot(
        data=plotDF,
        x=cell1 + " " + receptor1 + " " + treatment1,
        y=cell2 + " " + receptor2 + " " + treatment2,
        ax=ax,
        hue="Status",
    )


def plot_by_cell(sigDF, receptor1, treatment1, receptor2, treatment2, ax):
    """Plots receptor in pop 1 vs receptor in pop 2 per patient, by disease status"""
    plotDF = pd.DataFrame({"Cell": sigDF.loc[(sigDF.Treatment == treatment1)].index.values})
    plotDF[receptor1 + " " + treatment1] = sigDF.loc[(sigDF.Treatment == treatment1)][receptor1].values
    plotDF[receptor2 + " " + treatment2] = sigDF.loc[(sigDF.Treatment == treatment2)][receptor2].values
    sns.scatterplot(
        data=plotDF,
        x=receptor1 + " " + treatment1,
        y=receptor2 + " " + treatment2,
        ax=ax,
        hue="Cell",
    )


def plot_per_patient(sigDF, receptor1, treatment1, receptor2, treatment2, ax):
    """Plots receptor in pop 1 vs receptor in pop 2 per patient, by disease status"""
    status_DF = get_status_df()
    plotDF = pd.DataFrame({"Patient": sigDF.loc[(sigDF.Treatment == treatment1)].index.values})
    plotDF[receptor1 + " " + treatment1] = sigDF.loc[(sigDF.Treatment == treatment1)][receptor1].values
    plotDF[receptor2 + " " + treatment2] = sigDF.loc[(sigDF.Treatment == treatment2)][receptor2].values
    plotDF = plotDF.set_index("Patient").join(
        status_DF.set_index("Patient"), on="Patient"
    )
    sns.scatterplot(
        data=plotDF,
        x=receptor1 + " " + treatment1,
        y=receptor2 + " " + treatment2,
        ax=ax,
        hue="Status",
    )