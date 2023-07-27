"""
This creates Figure 1.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from copy import copy
import tensorly as tl
from .common import subplotLabel, getSetup, path_here
from os.path import join
from ..tensor import factorTensor, get_status_dict


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 3), multz={0: 2})
    # make_flow_sc_dataframe()
    # make_flow_sc_dataframe_rec()

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    CoH_data = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    CoH_data_I = make_impute_DF()
    treatments = ["IL2-50ng", "IL4-50ng", "IL6-50ng", "IL10-50ng", "IFNg-50ng", "TGFB-50ng", "IFNg-50ng+IL6-50ng", "Untreated"]
    CoH_data = CoH_data.loc[CoH_data.Treatment.isin(treatments)]
    fullHeatMap(ax[1], CoH_data, CoH_data_I, ["pSTAT3"], cbar=False, makeDF=False)
    fullHeatMap(ax[2], CoH_data, CoH_data_I, ["pSTAT5"], cbar=False, makeDF=False)
    fullHeatMap(ax[3], CoH_data, CoH_data_I, ["pSTAT6"], cbar=False, makeDF=False)

    return f


def fullHeatMap(ax, respDF, respDF_I, markers, cbar=True, makeDF=True):
    """Plots the various affinities for IL-2 Muteins"""
    heatmapDF = pd.DataFrame()
    respDFhm = copy(respDF)
    respDFhm = respDFhm.groupby(["Patient", "Cell", "Time", "Treatment", "Marker"]).Mean.mean().reset_index()
    respDFhm_I = copy(respDF_I)
    respDFhm_I = respDFhm_I.groupby(["Patient", "Cell", "Treatment", "Marker"]).Mean.mean().reset_index()
    patients = get_status_dict().keys()
    if makeDF:
        for cell in respDFhm.Cell.unique():
            print(cell)
            for patient in patients:
                row = pd.DataFrame()
                row["Patient/Cell"] = [patient + " - " + str(cell)]
                for treatment in respDFhm.Treatment.unique():
                    normMax = respDFhm.loc[(respDFhm.Patient == patient) & (respDFhm.Cell == cell)].Mean.max()
                    for time in ["15min"]:
                        for marker in markers:
                            entry = respDFhm.loc[(respDFhm.Patient == patient) & (respDFhm.Treatment == treatment) & (respDFhm.Cell == cell)
                                                 & (respDFhm.Time == time) & (respDFhm.Marker == marker)].Mean.values / normMax
                            if np.isnan(entry) or entry.size < 1:
                                entry = respDFhm_I.loc[(respDFhm_I.Patient == patient) & (respDFhm_I.Treatment == treatment) & (respDFhm_I.Cell == cell)
                                                       & (respDFhm_I.Marker == marker)].Mean.values / normMax
                            row[treatment + " - " + str(time)] = entry
                heatmapDF = pd.concat([heatmapDF, row])
        heatmapDF.to_csv(join(path_here, "data/CoH_Heatmap_DF_" + markers[0] + ".csv"))
    else:
        heatmapDF = pd.read_csv(join(path_here, "data/CoH_Heatmap_DF_" + markers[0] + ".csv"), index_col=0)
    heatmapDF = heatmapDF.set_index("Patient/Cell")
    sns.heatmap(data=heatmapDF, vmin=0, vmax=1, ax=ax, cbar=cbar, cbar_kws={'label': markers[0]}, yticklabels=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


def make_impute_DF():
    """Imputes data and returns df containing those values"""
    num_comps = 5
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Tensor_DataSet.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, r=num_comps)
    tensor = tl.cp_to_tensor(tFacAllM)
    CoH_Data_I = copy(CoH_Data)
    CoH_Data_I.data = tensor
    CoH_Data_DF = CoH_Data_I.to_dataframe(name=["Tensor"])
    CoH_Data_DF = CoH_Data_DF.reset_index()
    CoH_Data_DF.columns = CoH_Data_DF.columns.map(''.join)
    CoH_Data_DF = CoH_Data_DF.rename(columns={"Tensor": "Mean"})

    return CoH_Data_DF
