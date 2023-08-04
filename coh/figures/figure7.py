"""
This creates Figure S2, factorization of fold-change data.
"""
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from .common import subplotLabel, getSetup, BC_scatter_cells
from ..flow_rec import get_status_rec_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 7), (2, 4))

    # Add subplot labels
    subplotLabel(ax)
    CoH_DF = pd.read_csv("./coh/data/CoH_Flow_DF.csv", index_col=0)
    treatments = np.array(["IL2-50ng", "IL4-50ng", "IL6-50ng", "IL10-50ng", "IFNg-50ng", "TGFB-50ng", "IFNg-50ng+IL6-50ng", "Untreated"])
    CoH_DF = CoH_DF.loc[CoH_DF.Treatment.isin(treatments)].dropna()

    # Figure A Markers for signaling component

    # B Baseline pSTAT3

    BC_scatter_cells(ax[1], CoH_DF, "pSTAT3", "Untreated")

    # C Baseline pSTAT3 vs pSTAT3 induced

    meanDF = CoH_DF.groupby(["Patient", "Cell", "Treatment", "Marker"]).mean().reset_index()

    meanDF = (meanDF.pivot(index=["Patient", "Cell", "Treatment"], columns="Marker", values="Mean").reset_index().set_index("Patient"))
    print(meanDF)
    meanDF.iloc[:, 2::] = meanDF.iloc[:, 2::].apply(zscore)
    print(meanDF)

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

    # C CD8+ pSTAT3 vs B pSTAT3 in IL10

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

    return f


def plot_by_patient(sigDF, cell1, receptor1, treatment1, cell2, receptor2, treatment2, ax):
    """Plots receptor in pop 1 vs receptor in pop 2 per patient, by disease status"""
    status_DF = get_status_rec_df()
    plotDF = pd.DataFrame({"Patient": sigDF.loc[(sigDF.Cell == cell1) & (sigDF.Treatment == treatment1)].index.values})
    plotDF[cell1 + " " + receptor1 + " " + treatment1] = sigDF.loc[(sigDF.Cell == cell1) & (sigDF.Treatment == treatment1)][receptor1].values
    plotDF[cell2 + " " + receptor2 + " " + treatment2] = sigDF.loc[(sigDF.Cell == cell2) & (sigDF.Treatment == treatment2)][receptor2].values
    plotDF = plotDF.set_index("Patient").join(
        status_DF.set_index("Patient"), on="Patient"
    )
    print(plotDF.corr())
    sns.scatterplot(
        data=plotDF,
        x=cell1 + " " + receptor1 + " " + treatment1,
        y=cell2 + " " + receptor2 + " " + treatment2,
        ax=ax,
        hue="Status",
    )
