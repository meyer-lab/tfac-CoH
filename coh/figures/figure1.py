"""
This creates Figure 1.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from copy import copy
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import cp_normalize, perform_CP
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (4, 2), multz={0: 1, 2: 3})

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    marker = "pSTAT3"
    CoH_data = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    fullHeatMap(ax[1], CoH_data, [marker], makeDF=False)
    response_ligand_scatter(ax[2], CoH_data, "pSTAT5")
    response_cell_scatter(ax[3], CoH_data, "pSTAT6", "IL4-50ng")

    return f


def fullHeatMap(ax, respDF, markers, makeDF=True):
    """Plots the various affinities for IL-2 Muteins"""
    heatmapDF = pd.DataFrame()
    respDFhm = copy(respDF)
    respDFhm = respDFhm.groupby(["Patient", "Cell", "Time", "Treatment", "Marker"]).Mean.mean().reset_index()
    patients = [
        "Patient 26",
        "Patient 28",
        "Patient 30",
        "Patient 34",
        "Patient 35",
        "Patient 43",
        "Patient 44",
        "Patient 45",
        "Patient 52",
        "Patient 52A",
        "Patient 54",
        "Patient 56",
        "Patient 58",
        "Patient 60",
        "Patient 61",
        "Patient 62",
        "Patient 63",
        "Patient 66",
        "Patient 68",
        "Patient 69",
        "Patient 70",
        "Patient 79",
        "Patient 4",
        "Patient 8",
        "Patient 406",
        "Patient 10-T1",
        "Patient 10-T2",
        "Patient 10-T3",
        "Patient 15-T1",
        "Patient 15-T2",
        "Patient 15-T3",
        "Patient 19186-2",
        "Patient 19186-3",
        "Patient 19186-14",
        "Patient 21368-3",
        "Patient 21368-4"]
    if makeDF:
        for cell in respDFhm.Cell.unique():
            print(cell)
            for patient in patients:
                row = pd.DataFrame()
                row["Patient/Cell"] = [patient + " - " + str(cell)]
                for treatment in respDFhm.Treatment.unique():
                    normMax = respDFhm.loc[(respDFhm.Patient == patient) & (respDFhm.Cell == cell)].Mean.max()
                    for time in respDF.Time.unique():
                        for marker in markers:
                            entry = respDFhm.loc[(respDFhm.Patient == patient) & (respDFhm.Treatment == treatment) & (respDFhm.Cell == cell)
                                                 & (respDFhm.Time == time) & (respDFhm.Marker == marker)].Mean.values / normMax
                            if np.isnan(entry):
                                row[treatment + " - " + str(time) + " hrs"] = np.nan
                            elif entry.size < 1:
                                row[treatment + " - " + str(time) + " hrs"] = np.nan
                            else:
                                row[treatment + " - " + str(time) + " hrs"] = entry
                heatmapDF = pd.concat([heatmapDF, row])
        heatmapDF = heatmapDF.set_index("Patient/Cell")
        heatmapDF.to_csv(join(path_here, "data/CoH_Heatmap_DF.csv"))
    else:
        heatmapDF = pd.read_csv(join(path_here, "data/CoH_Heatmap_DF.csv"), index_col=0)

    sns.heatmap(data=heatmapDF, vmin=0, vmax=1, ax=ax, cbar_kws={'label': markers[0]})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


def response_ligand_scatter(ax, CoH_DF, marker, cells=False):
    """Scatters specific responses"""
    hist_DF = CoH_DF.loc[(CoH_DF.Marker == marker)]
    hist_DF = hist_DF.groupby(["Treatment", "Patient"]).Mean.mean().reset_index()

    sns.boxplot(data=hist_DF, y="Mean", x="Treatment", ax=ax)
    ax.set(title="Average " + marker + " Response", ylabel=marker, xlabel="Treament")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


def response_cell_scatter(ax, CoH_DF, marker, treatment):
    """Scatters specific responses"""
    hist_DF = CoH_DF.loc[(CoH_DF.Marker == marker) & (CoH_DF.Treatment == treatment)]
    hist_DF = hist_DF.groupby(["Cell", "Patient"]).Mean.mean().reset_index()

    sns.boxplot(data=hist_DF, y="Mean", x="Cell", ax=ax)
    ax.set(title="Average Response to " + treatment, ylabel=marker, xlabel="Cell")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")