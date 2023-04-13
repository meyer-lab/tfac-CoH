"""
This creates Figure 4.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
import scanpy as sc
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..imports import makeRNAseqDF, importCITE, importRNACITE, makeRNAseqDF_Ann
import matplotlib.pyplot as plt


plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)
    makeRNAseqDF_Ann(surface=True)
    makeRNAseqDF_Ann(surface=False)
    

    CITE_DF = importCITE()
    CITE_DF_R = importRNACITE()

    CD8_DF = CITE_DF.loc[CITE_DF.CellType2.isin(["CD8 TCM", "CD8 TEM", "CD8 Naive"]), :]
    CD8_DF_R = CITE_DF_R[np.isin(CITE_DF_R.obs.CellType2, ["CD8 TCM", "CD8 TEM", "CD8 Naive"]), :]
    CD8_DF_2 = CD8_DF_R.to_df()["CD274"].to_frame()
    CD8_DF_2.columns = ["CD274"]

    B_DF = CITE_DF.loc[CITE_DF.CellType1.isin(["B"])]
    B_DF_R = CITE_DF_R[np.isin(CITE_DF_R.obs.CellType1, ["B"]), :]
    B_DF_2 = B_DF_R.to_df()["CD274"].to_frame()
    B_DF_2.columns = ["CD274"]

    sns.histplot(data=CD8_DF, x="CD274", ax=ax[0])
    ax[0].set(title="CD8 Cells, Surface Data", xlim=(0, 50))
    sns.histplot(data=CD8_DF_2, x="CD274", ax=ax[1])
    ax[1].set(title="CD8 Cells, RNA Data", xlim=(0, 5))

    sns.histplot(data=B_DF, x="CD274", ax=ax[2])
    ax[2].set(title="B Cells, Surface Data", xlim=(0, 50))
    sns.histplot(data=B_DF_2, x="CD274", ax=ax[3])
    ax[3].set(title="B Cells, RNA Data", xlim=(0, 5))    

    """
    sc.pp.pca(CITE_DF_R, svd_solver='arpack')
    sc.pp.neighbors(CITE_DF_R)
    sc.tl.leiden(CITE_DF_R, resolution=0.75)
    sc.tl.umap(CITE_DF_R)
    sc.pp.subsample(CITE_DF_R, fraction=0.1, random_state=0)
    sc.pl.umap(CITE_DF_R, color='leiden', legend_loc='on data', title='', frameon=False, ax=ax[0])
    sc.pl.umap(CITE_DF_R, color='CellType1', legend_loc='on data', title='', frameon=False, ax=ax[1])
    sc.pl.umap(CITE_DF_R, color='CellType2', legend_loc='on data', title='', frameon=False, ax=ax[2])
    sc.pl.umap(CITE_DF_R, color='CellType3', legend_loc='on data', title='', frameon=False, ax=ax[3])
    """

    return f
