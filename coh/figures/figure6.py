"""
This creates Figure 4.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
import scanpy as sc
import anndata as ad
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..imports import importCITE, importRNACITE, makeCITE_Ann, makeAzizi_Ann, process_Azizi, import_Azizi_Unfilt
import matplotlib.pyplot as plt


plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 3))

    # Add subplot labels
    subplotLabel(ax)
    # makeRNAseqDF_Ann(surface=True)
    # makeRNAseqDF_Ann(surface=False)
    # process_Azizi()


    # CITE_DF = importCITE()
    CITE_DF_R = importRNACITE()
    sc.pp.normalize_total(CITE_DF_R, target_sum=1e4)
    sc.pp.log1p(CITE_DF_R)
    sc.pp.scale(CITE_DF_R, max_value=10)
    sc.pl.violin(CITE_DF_R, ['TFRC'], groupby='CellType1', ax=ax[0])
    # CITE_DF_R = CITE_DF_R[CITE_DF_R.obs.CellType1 == "B", :]
    # CITE_DF_R.obs["Patient"] = "Healthy"
    """

    CD8_DF = CITE_DF.loc[CITE_DF.CellType2.isin(["CD8 TCM", "CD8 TEM", "CD8 Naive"]), :]
    CD8_DF_R = CITE_DF_R[np.isin(CITE_DF_R.obs.CellType2, ["CD8 TCM", "CD8 TEM", "CD8 Naive"]), :]
    CD8_DF_2 = CD8_DF_R.to_df()["CD274"].to_frame()
    CD8_DF_2.columns = ["CD274"]

    B_DF = CITE_DF.loc[CITE_DF.CellType1.isin(["B"])]
    B_DF_R = CITE_DF_R[np.isin(CITE_DF_R.obs.CellType1, ["B"]), :]
    B_DF_2 = B_DF_R.to_df()["CD274"].to_frame()
    B_DF_2.columns = ["CD274"]
    print(B_DF_2)

    sns.histplot(data=CD8_DF, x="CD274", ax=ax[0])
    ax[0].set(title="CD8 Cells, Surface Data", xlim=(0, 50))
    sns.histplot(data=CD8_DF_2, x="CD274", ax=ax[1])
    ax[1].set(title="CD8 Cells, RNA Data", xlim=(0, 5))

    sns.histplot(data=B_DF, x="CD274", ax=ax[2])
    ax[2].set(title="B Cells, Surface Data", xlim=(0, 50))
    sns.histplot(data=B_DF_2, x="CD274", ax=ax[3])
    ax[3].set(title="B Cells, RNA Data", xlim=(0, 5))
    """

    RNA1 = import_Azizi_Unfilt(patient="BC01")
    # RNA1 = RNA1[RNA1.obs.leiden == "B", :]
    # RNA.layers['scaled'] = sc.pp.scale(RNA1, copy=True).X
    # RNA = RNA1.to_df()
    # sns.histplot(data=RNA1, x="CD274", ax=ax[4])
    # ax[4].set(title="B Cells, Patient 1", xlim=(-0.2, 0.5))
    sc.pl.violin(RNA1, ['TFRC'], groupby='leiden', ax=ax[1])
    # RNA1.obs["Patient"] = "BC01"
    # RNA1 = RNA1[RNA1.obs.leiden == "B", :]

    RNA4 = import_Azizi_Unfilt(patient="BC04")
    # RNA4 = RNA4[RNA4.obs.leiden == "B", :]
    # RNA4 = RNA4.to_df()
    # sns.histplot(data=RNA4, x="CD274", ax=ax[5])
    # ax[5].set(title="B Cells, Patient 4", xlim=(-0.2, 0.5))
    # RNA4.layers['scaled'] = sc.pp.scale(RNA4, copy=True).X
    sc.pl.violin(RNA4, ['TFRC'], groupby='leiden', ax=ax[2])
    # RNA4.obs["Patient"] = "BC04"
    # RNA4 = RNA4[RNA4.obs.leiden == "B", :]

    #all_RNA = ad.concat([CITE_DF_R, RNA1, RNA4])
    #marker_genes_dict = {
    #'Patient': ['IL2RA', 'TFRC', 'CD274']}
    #sc.pl.matrixplot(all_RNA, marker_genes_dict, 'Patient', dendrogram=True, colorbar_title='mean z-score', vmin=-0.5, vmax=0.5, cmap='RdBu_r', ax=ax[0])
    


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