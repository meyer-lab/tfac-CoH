"""
This creates Figure S2, factorization of fold-change data.
"""
from ..tensor import (
    factorTensor,
    R2Xplot,
    CoH_LogReg_plot,
    BC_status_plot,
)
from ..flow import make_CoH_Tensor
from .common import subplotLabel, getSetup, BC_scatter_cells_rec
from ..flow_rec import get_status_rec_df
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (2, 4))

    # Add subplot labels
    subplotLabel(ax)
    CoH_DF = pd.read_csv("./coh/data/CoH_Rec_DF.csv", index_col=0)

    # Figure A - plot of PD-1 CD8 Cells

    DF = CoH_DF.loc[CoH_DF.Marker == "PD1"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    PD1_DF = DF.loc[DF.Cell.isin(["CD8+", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 Naive", "CD8 TEMRA"])]
    BC_scatter_cells_rec(ax[0], PD1_DF, "PD1", filter=False)
    

    # B PD-L1 CD8 and B cells

    DF = CoH_DF.loc[CoH_DF.Marker == "PD_L1"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    PDL1_DF = DF.loc[DF.Cell.isin(["CD8+", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 Naive", "CD8 TEMRA", "CD20 B", "CD20 B Naive", "CD20 B Memory"])]
    BC_scatter_cells_rec(ax[1], PDL1_DF, "PD_L1", filter=False)

    # C IL6Ra B

    DF = CoH_DF.loc[CoH_DF.Marker == "IL6Ra"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    IL6Ra_DF = DF.loc[DF.Cell.isin(["CD8+", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 Naive", "CD8 TEMRA", "CD20 B", "CD20 B Naive", "CD20 B Memory"])]
    BC_scatter_cells_rec(ax[2], IL6Ra_DF, "IL6Ra", filter=False)

    # D IL2Ra Tregs

    DF = CoH_DF.loc[CoH_DF.Marker == "IL2Ra"]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    IL2Ra_DF = DF.loc[DF.Cell.isin(["Treg"])]
    BC_scatter_cells_rec(ax[3], IL2Ra_DF, "IL2Ra", filter=False)

    # Make mean Z scored DF
    meanDF = CoH_DF.groupby(["Patient", "Cell", "Marker"]).mean().reset_index() 
    
    meanDF = meanDF.pivot(index=['Patient', "Cell"], columns='Marker', values='Mean').reset_index().set_index("Patient")
    meanDF.loc[:, meanDF.columns.values != "Cell"] = meanDF.loc[:, meanDF.columns.values != "Cell"].apply(zscore)



    # E PD-L1 in B vs CD8 Cells

    plot_by_patient(meanDF, cell1="CD8 TEM", receptor1="PD_L1", cell2="CD20 B", receptor2="PD_L1", ax=ax[4])

    # F IL6Ra in B vs CD8 Cells

    plot_by_patient(meanDF, cell1="CD8+", receptor1="IL6Ra", cell2="CD20 B", receptor2="IL6Ra", ax=ax[5])

    # G IL2Ra Tregs vs PD-L1 CD8s

    plot_by_patient(meanDF, cell1="Treg", receptor1="IL2Ra", cell2="CD8 TEM", receptor2="PD_L1", ax=ax[6])

    # H IL2Ra Tregs vs IL6Ra B

    plot_by_patient(meanDF, cell1="Treg", receptor1="IL2Ra", cell2="CD20 B", receptor2="IL6Ra", ax=ax[7])

    # I Univariate vs coordinated ROC

    return f


def plot_by_patient(recDF, cell1, receptor1, cell2, receptor2, ax):
    """Plots receptor in pop 1 vs receptor in pop 2 per patient, by disease status"""
    status_DF = get_status_rec_df()
    plotDF = pd.DataFrame({"Patient": recDF.loc[recDF.Cell == cell1].index.values})
    plotDF[cell1 + " " + receptor1] = recDF.loc[recDF.Cell == cell1][receptor1].values
    plotDF[cell2 + " " + receptor2] = recDF.loc[recDF.Cell == cell2][receptor2].values
    plotDF = plotDF.set_index("Patient").join(status_DF.set_index("Patient"), on="Patient")
    sns.scatterplot(data=plotDF, x=cell1 + " " + receptor1, y=cell2 + " " + receptor2, hue="Status", ax=ax)
