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
    ax, f = getSetup((16, 8), (2, 4))

    # Add subplot labels
    subplotLabel(ax)
    RNA = import_Azizi(patient="BC01")
   
    sc.pp.pca(RNA, svd_solver='arpack')
    sc.pp.neighbors(RNA)
    sc.tl.leiden(RNA, resolution=0.6)
    sc.tl.umap(RNA)
    sc.tl.rank_genes_groups(RNA, groupby='leiden', method='wilcoxon')
    marker_matches = sc.tl.marker_gene_overlap(RNA, marker_genes)
    print(marker_matches)
    sc.pl.umap(RNA, color=['batch'], ax=ax[1])
    sc.pl.umap(RNA, color=['CD19'], ax=ax[2])
    sc.pl.umap(RNA, color=['CD8B'], ax=ax[3])
    new_cluster_names = ['T', 'NK/Cytotoxic', 'Monocyte', 'B']
    RNA.rename_categories('leiden', new_cluster_names)
    sc.pl.umap(RNA, color='leiden', legend_loc='on data', title='', frameon=False, ax=ax[0])
    RNA.write_h5ad("/opt/CoH/SingleCell/Patient_BC01/BC01_processed_annot.h5ad.gz", compression='gzip')

    RNA = import_Azizi(patient="BC04")
    sc.pp.pca(RNA, svd_solver='arpack')
    sc.pp.neighbors(RNA)
    sc.tl.leiden(RNA, resolution=0.6)
    sc.tl.umap(RNA)
    
    sc.tl.rank_genes_groups(RNA, groupby='leiden', method='wilcoxon')
    marker_matches = sc.tl.marker_gene_overlap(RNA, marker_genes)
    print(marker_matches)
    sc.pl.umap(RNA, color=['batch'], ax=ax[5])
    sc.pl.umap(RNA, color=['CD19'], ax=ax[6])
    sc.pl.umap(RNA, color=['CD8B'], ax=ax[7])
    new_cluster_names = ['T', 'NK/Cytotoxic', 'B', 'None', 'Monocytes', 'DC']
    RNA.rename_categories('leiden', new_cluster_names)
    sc.pl.umap(RNA, color='leiden', legend_loc='on data', title='', frameon=False, ax=ax[4])
    RNA.write_h5ad("/opt/CoH/SingleCell/Patient_BC04/BC04_processed_annot.h5ad.gz", compression='gzip')


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
        'CD8B'],
    'Mast Cells': ['ENPP3', 'KIT'], 
    'plasmacytoid DC': ['IL3RA', 'LILRA4']
}