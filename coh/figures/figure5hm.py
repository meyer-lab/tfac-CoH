"""
This creates Figure 4.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from tensorpack.cmtf import cp_normalize
from .figureCommon import subplotLabel, getSetup, BC_scatter, BC_scatter_cells
from os.path import join, dirname
from ..flow import make_flow_df, make_CoH_Tensor
from ..tensor import get_status_dict
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 7), (1, 1))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df(foldChange=True)
    # make_CoH_Tensor(just_signal=True, foldChange=True)

    #make_alldata_DF(CoH_Data, PCA=False, foldChange=True)
    CoH_DF = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    CoH_DF_B = pd.read_csv(join(path_here, "data/CoH_Flow_DF_Basal.csv"))
    CoH_Data_DF_R = pd.read_csv(join(path_here, "data/CoH_Rec_DF.csv"))

    # CD8
    # f = dysreg_cor_hm_R(CoH_DF, CoH_DF_B, CoH_Data_DF_R, ["pSTAT3", "pSTAT5", "pSmad1-2"], ["IL10-50ng", "IL2-50ng", "TGFB-50ng"], ["pSmad1-2", "pSTAT4"], ["TGFB RII", "PD_L1", "IL6Ra", "IL10R", "IL12RI", "IL2RB"], cells=["CD8+"])
    # CD4
    f = dysreg_cor_hm_R(CoH_DF, CoH_DF_B, CoH_Data_DF_R, ["pSTAT3", "pSmad1-2"], ["IL10-50ng", "TGFB-50ng"], ["pSmad1-2", "pSTAT4", "pSTAT1"], ["TGFB RII", "IL10R", "IL6Ra", "IL12RI", "IFNg R1"], cells=["CD4+"])
    # Bcell
    # f = dysreg_cor_hm_R(CoH_DF, CoH_DF_B, CoH_Data_DF_R, ["pSTAT3", "pSTAT5", "pSmad1-2"], ["IL10-50ng", "IL2-50ng", "TGFB-50ng"], ["pSmad1-2", "pSTAT4"], ["TGFB RII", "PD_L1", "IL6Ra", "IL10R", "IL12RI", "IL2RB", "IL2Ra"], cells=["CD20 B"])
    # Treg
    # f = dysreg_cor_hm_R(CoH_DF, CoH_DF_B, CoH_Data_DF_R, ["pSTAT3", "pSTAT5", "pSmad1-2"], ["IL10-50ng", "IL2-50ng", "TGFB-50ng"], ["pSmad1-2", "pSTAT4", "pSTAT1"], ["TGFB RII", "PD_L1", "IL6Ra", "IL10R", "IL12RI", "IL2RB", "IL2Ra", "IFNg R1"], cells=["Treg"])
    # Monocytes
    # f = dysreg_cor_hm_R(CoH_DF, CoH_DF_B, CoH_Data_DF_R, ["pSTAT3", "pSTAT5", "pSmad1-2"], ["IL10-50ng", "IL2-50ng", "TGFB-50ng"], ["pSmad1-2", "pSTAT4"], ["TGFB RII", "PD_L1", "IL6Ra", "IL10R", "IL12RI", "IL2RB"], cells=["Classical Monocyte"])

    return f


def dysreg_cor_hm(CoH_DF, CoH_DF_B, markers_dysreg, cyto_dysreg, markers_dysreg_B, ax, cells=False):
    """Plots possible correlation of dysregulation"""
    CoH_DF = CoH_DF.groupby(["Cell", "Patient", "Treatment", "Marker"]).Mean.mean().reset_index()
    CoH_DF_B = CoH_DF_B.groupby(["Cell", "Patient", "Treatment", "Marker"]).Mean.mean().reset_index()
    dysreg_DF = pd.DataFrame()
    if type(cells) == list:
        CoH_DF = CoH_DF.loc[CoH_DF.Cell.isin(cells)]
        CoH_DF_B = CoH_DF_B.loc[CoH_DF_B.Cell.isin(cells)]
    for patient in CoH_DF.Patient.unique():
        patient_DF = CoH_DF.loc[CoH_DF.Patient == patient]
        patient_row = pd.DataFrame()
        for i, marker in enumerate(markers_dysreg):
            value = patient_DF.loc[(patient_DF.Marker == marker) & (patient_DF.Treatment == cyto_dysreg[i])].Mean.values
            patient_row[markers_dysreg[i] + " response to " + cyto_dysreg[i]] = value

        patient_DF_B = CoH_DF_B.loc[CoH_DF_B.Patient == patient]
        for i, marker in enumerate(markers_dysreg_B):
            value = patient_DF_B.loc[(patient_DF_B.Marker == marker) & (patient_DF_B.Treatment == "Untreated")].Mean.values
            patient_row["Basal " + marker] = value
        dysreg_DF = pd.concat([dysreg_DF, patient_row])
    print(dysreg_DF)
    f = sns.heatmap(data=dysreg_DF.corr(), vmin=-1, vmax=1, annot=True, ax=ax)
    #f = sns.clustermap(data=dysreg_DF.corr(), robust=True, vmin=-1, vmax=1, row_cluster=False, figsize=(8, 3))
    return f


def dysreg_cor_hm_R(CoH_DF, CoH_DF_B, CoH_DF_R, markers_dysreg, cyto_dysreg, markers_dysreg_B, markers_dysreg_R, cells=False):
    """Plots possible correlation of dysregulation"""
    CoH_DF = CoH_DF.loc[CoH_DF.Patient != "Patient 406"]
    CoH_DF_B = CoH_DF_B.loc[CoH_DF_B.Patient != "Patient 406"]
    CoH_DF_R = CoH_DF_R.loc[CoH_DF_R.Patient != "Patient 19186-12"]
    

    CoH_DF = CoH_DF.groupby(["Cell", "Patient", "Treatment", "Marker"]).Mean.mean().reset_index()
    CoH_DF_B = CoH_DF_B.groupby(["Cell", "Patient", "Treatment", "Marker"]).Mean.mean().reset_index()
    CoH_DF_R = CoH_DF_R.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()

    dysreg_DF = pd.DataFrame()
    status_DF = pd.read_csv(join(path_here, "data/Patient_Status.csv"))
    BC_Patients = status_DF.loc[status_DF.Status == "BC"].Patient.unique()
    CoH_DF = CoH_DF.loc[CoH_DF.Patient.isin(BC_Patients)]
    CoH_DF_B = CoH_DF_B.loc[CoH_DF_B.Patient.isin(BC_Patients)]
    CoH_DF_R = CoH_DF_R.loc[CoH_DF_R.Patient.isin(BC_Patients)]

    if type(cells) == list:
        CoH_DF = CoH_DF.loc[CoH_DF.Cell.isin(cells)]
        CoH_DF_B = CoH_DF_B.loc[CoH_DF_B.Cell.isin(cells)]
        CoH_DF_R = CoH_DF_R.loc[CoH_DF_R.Cell.isin(cells)]

    for patient in CoH_DF.Patient.unique():
        patient_DF = CoH_DF.loc[CoH_DF.Patient == patient]
        patient_row = pd.DataFrame()
        for i, marker in enumerate(markers_dysreg):
            value = patient_DF.loc[(patient_DF.Marker == marker) & (patient_DF.Treatment == cyto_dysreg[i])].Mean.values
            patient_row[markers_dysreg[i] + " response to " + cyto_dysreg[i]] = value

        patient_DF_B = CoH_DF_B.loc[CoH_DF_B.Patient == patient]
        for i, marker in enumerate(markers_dysreg_B):
            value = patient_DF_B.loc[(patient_DF_B.Marker == marker) & (patient_DF_B.Treatment == "Untreated")].Mean.values
            patient_row["Basal " + marker] = value

        patient_DF_R = CoH_DF_R.loc[CoH_DF_R.Patient == patient]
        for i, marker in enumerate(markers_dysreg_R):
            value = patient_DF_R.loc[(patient_DF_R.Marker == marker)].Mean.values
            patient_row[marker] = value
        dysreg_DF = pd.concat([dysreg_DF, patient_row])

    print(dysreg_DF)
    # f = sns.heatmap(data=dysreg_DF.corr(), vmin=-1, vmax=1, annot=True, ax=ax)
    cmap = sns.color_palette("vlag", as_cmap=True)
    f = sns.clustermap(data=dysreg_DF.corr(), robust=True, vmin=-1, vmax=1, row_cluster=True, col_cluster=True, annot=True, cmap=cmap, figsize=(8, 8))
    return f