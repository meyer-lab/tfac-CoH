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
from ..imports import importCITE, importRNACITE, makeCITE_Ann, makeAzizi_Ann, process_Azizi, import_Azizi
import matplotlib.pyplot as plt


plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)
    #makeRNAseqDF_Ann(surface=True)
    #makeRNAseqDF_Ann(surface=False)

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

    return f


marker_genes = {
    'Monocytes': [
        'CD14',
        'CD33',
        'LYZ',
        'LGALS3',
        'CSF1R',
        'ITGAX',
        'HLA-DRB1'],
    'Dendritic Cells': [
        'LAD1',
        'LAMP3',
        'TSPAN13',
        'CLIC2',
        'FLT3'],
    'B-cells': [
        'MS4A1',
        'CD19',
        'CD79A'],
    'T-helpers': [
        'TNF',
        'TNFRSF18',
        'IFNG',
        'IL2RA',
        'BATF'],
    'T cells': [
        'CD27',
        'CD69',
        'CD2',
        'CD3D',
        'CXCR3',
        'CCL5',
        'IL7R',
        'CXCL8',
        'GZMK'],
    'Natural Killers': [
        'NKG7',
        'GNLY',
        'PRF1',
        'FCGR3A',
        'NCAM1',
        'TYROBP'],
    'CD8': [
        'CD8A',
        'CD8B']
}