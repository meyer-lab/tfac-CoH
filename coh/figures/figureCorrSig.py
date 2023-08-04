"""
This creates Figure S2, factorization of fold-change data.
"""
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from .common import subplotLabel, getSetup, BC_scatter_cells
from ..flow import get_status_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 7), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    status = get_status_df()

    ddtype = {
        "Patient": "category",
        "Time": "category",
        "Treatment": "category",
        "Cell": "category",
        "Marker": "category",
    }
    CoH_DF = pd.read_csv("./coh/data/CoH_Flow_DF.csv", index_col=0, dtype=ddtype)

    # A, pSTAT3 response to IL-10 grouped by status
    BC_scatter_cells(ax[0], CoH_DF, "pSTAT3", "IL10-50ng")

    # (b) Baseline pSTAT3 in untreated cells across cell types, grouped by patient status.
    BC_scatter_cells(ax[1], CoH_DF, "pSTAT3", "Untreated")

    # (c) Baseline pSTAT3 versus IL-10 induced pSTAT3 in CD8-positive cells, across patients.
    # df = CoH_DF.loc[CoH_DF.Cell == "CD8+"]
    # df = df.loc[CoH_DF.Marker == "pSTAT3"]

    # (d) pSTAT3 response to IL-10 across patients in B cells versus CD8 TCM.

    # (e) Baseline untreated versus IL-2-induced STAT5 phosphorylation in CD8-positive cells.

    # (f) IL-2-induced pSTAT5 in CD8-positive cells and Tregs.

    # (g) Components 4 and 5 from the CPD factorization in Figure 2.
    with open('./coh/data/signaling.pkl', 'rb') as ff:
        tfac = pickle.load(ff).factors[0] # 12 component

    sns.scatterplot(ax=ax[6], x=tfac[:, 4], y=tfac[:, 5], hue=status.Status)

    # (h) The difference in average Smad1/2 phosphorylation between the BC and healthy cohorts versus the same quantity for STAT4 phosphorylation, plotted for each cell type.

    # (i–j) Baseline pSmad1/2 versus pSTAT4 across patients in CD8-positive cells (i) and Tregs (j).


    return f
