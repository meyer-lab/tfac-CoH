"""
This file includes various methods for flow cytometry analysis of fixed cells.
"""
from importlib.abc import PathEntryFinder
import os
from os.path import dirname, join
from pathlib import Path
import ast
import textwrap
from types import CellType
import pandas as pd
import numpy as np
import warnings
import xarray as xa
from copy import copy
from FlowCytometryTools import FCMeasurement
from .flow import pop_gate, live_PBMC_gate, pop_gate, get_gate_dict

path_here = os.path.dirname(os.path.dirname(__file__))


warnings.filterwarnings("ignore")
gate_df = pd.DataFrame()


marker_dict_1 = {"BUV661-A": "CD20",
                "BUV496-A": "CD14",
                "Alexa Fluor 700-A": "CD27",
                "APC-Cy7-A": "CD3",
                "BV750-A": "CD33",
                "BUV395-A": "CD45RA",
                "LIVE DEAD Blue-A": "Live/Dead",
                "BUV563-A": "CD4",
                "BUV737-A": "CD16",
                "BUV805-A": "CD8",
                "BB660-A": "IFNg R1",
                "BB700-A": "IL2Rgc",
                "PE-Cy7-A": "IL10R",
                "BV786-A": "IL2RB",
                "APC-A": "TGFB RII",
                "BV510-A": "PD_L1",
                "BV421-A": "IL12RI",
                "BV605-A": "PD1",
                "BB515-A": "IL2Ra",
                "BB630-A": "IL4Ra",
                "BB796-A": "IL6RB",
                "PE-A": "IL15Ra",
                "PE-CF594-A": "IL6Ra",
                "BV650-A": "IL7Ra"}

marker_dict_2 = {"BUV661-A": "CD20",
                "BUV496-A": "CD14",
                "Alexa Fluor 700-A": "CD27",
                "APC-Cy7-A": "CD3",
                "BV750-A": "CD33",
                "BUV395-A": "CD45RA",
                "LIVE DEAD Blue-A": "Live/Dead",
                "BUV563-A": "CD4",
                "APC-Fire 810-A": "CD16",
                "BUV805-A": "CD8",
                "BB660-A": "IFNg R1",
                "BB700-A": "IL2Rgc",
                "PE-Cy7-A": "IL10R",
                "BV786-A": "IL2RB",
                "APC-A": "TGFB RII",
                "BV510-A": "PD_L1",
                "BV421-A": "IL12RI",
                "BV605-A": "PD1",
                "BB515-A": "IL2Ra",
                "PE-A": "IL4Ra",
                "BUV737-A": "IL6RB",
                "PE-CF594-A": "IL6Ra",
                "BV650-A": "IL7Ra"}

marker_dict_3 = {"BUV661-A": "CD20",
                "BUV496-A": "CD14",
                "Alexa Fluor 700-A": "CD27",
                "APC-Cy7-A": "CD3",
                "BV750-A": "CD33",
                "BUV395-A": "CD45RA",
                "Alexa 350-A": "Live/Dead",
                "BUV563-A": "CD4",
                "BUV737-A": "CD16",
                "BUV805-A": "CD8",
                "BB660-A": "IFNg R1",
                "BB700-A": "IL2Rgc",
                "PE-Cy7-A": "IL10R",
                "BV786-A": "IL2RB",
                "APC-A": "TGFB RII",
                "BV510-A": "PD_L1",
                "BV421-A": "IL12RI",
                "BV605-A": "PD1",
                "BB515-A": "IL2Ra",
                "BB630-A": "IL4Ra",
                "BB796-A": "IL6RB",
                "PE-A": "IL15Ra",
                "PE-CF594-A": "IL6Ra",
                "BV650-A": "IL7Ra"}

panelDict = {"marker_dict_1": marker_dict_1,
            "marker_dict_2": marker_dict_2,
            "marker_dict_3": marker_dict_3}


def compile_patient_rec(patient_num):
    """Adds all data from a single patient to an FC file"""
    pathname = "/opt/CoH/Receptor Data/F" + patient_num.split(" ")[1] + "_Unmixed.fcs"
    FCfiles = []
    FCfiles.append(FCMeasurement(ID=patient_num, datafile=pathname))
    return combineWells_rec(FCfiles, patient_num.split(" ")[1])


def combineWells_rec(samples, patientNum):
    """Accepts sample array returned from importF, and array of channels, returns combined well data"""
    markerKey = pd.read_csv("coh/data/Patient_Receptor_Panels.csv")
    dictionary = markerKey.loc[markerKey.Patient == patientNum].Panel.values
    marker_dict = panelDict[dictionary[0]]
    markers = np.array(list(marker_dict.keys()))
    log_markers = markers[np.isin(markers, samples[0].data.columns)]
    samples = samples[0].transform("tlog", channels=log_markers)
    combinedSamples = samples
    combinedSamples.data = combinedSamples.data.rename(marker_dict, axis=1)
    combinedSamples.data = combinedSamples.data.astype(int)
    return combinedSamples


def process_sample_rec(sample, marker_dict):
    """Relabels and logs a sample"""
    markers = np.array(list(marker_dict.keys()))
    log_markers = markers[np.isin(markers, sample.data.columns)]
    sample = sample.transform("tlog", channels=log_markers)
    sample.data = sample.data.rename(marker_dict, axis=1)
    return sample, log_markers


gate_dict = get_gate_dict()


def make_flow_df_rec(subtract=True, abundance=False, foldChange=False):
    """Compiles data for all populations for all patients into .csv"""
    patients = ["Patient 26", "Patient 28", "Patient 30", "Patient 34", "Patient 35", "Patient 43", "Patient 44", "Patient 45", "Patient 52", "Patient 52A", "Patient 54", "Patient 56", "Patient 58", "Patient 60", "Patient 61", "Patient 62", "Patient 63", "Patient 66", "Patient 68", "Patient 69", "Patient 70", "Patient 79", "Patient 4", "Patient 8", "Patient 10-T1",  "Patient 10-T2",  "Patient 10-T3", "Patient 15-T1", "Patient 15-T2", "Patient 15-T3", "Patient 19186-2", "Patient 19186-3", "Patient 19186-12", "Patient 19186-14", "Patient 21368-3", "Patient 21368-4"]
    cell_types = ["T", "CD16 NK", "CD8+", "CD4+", "CD4-/CD8-", "Treg", "Treg 1", "Treg 2", "Treg 3", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 TEMRA",
                  "CD4 TEM", "CD4 TCM", "CD4 Naive", "CD4 TEMRA", "CD20 B", "CD20 B Naive", "CD20 B Memory", "CD33 Myeloid", "Classical Monocyte", "NC Monocyte"]
    gateDF = pd.read_csv(join(path_here, "coh/data/CoH_Flow_Gates_Receptors.csv")).reset_index().drop("Unnamed: 0", axis=1)
    CoH_DF_rec = pd.DataFrame([])
    markerKey = pd.read_csv(join(path_here, "coh/data/Patient_Receptor_Panels.csv"))

    for patient in patients:
        print(patient)
        patient_num = patient.split(" ")[1]
        dictionary = markerKey.loc[markerKey.Patient == patient_num].Panel.values
        marker_dict = panelDict[dictionary[0]]
        sample = FCMeasurement(ID="Sample", datafile="/opt/CoH/Receptor Data/F" + patient_num + "_Unmixed.fcs")
        sample, markers = process_sample_rec(sample, marker_dict)
        sample = live_PBMC_gate(sample, patient, gateDF)
        for cell_type in cell_types:
            pop_sample, _ = pop_gate(sample, cell_type, patient, gateDF)
            for marker in markers:
                mean = pop_sample.data[marker_dict[marker]]
                mean = np.mean(mean.values[mean.values < np.quantile(mean.values, 0.995)])
                CoH_DF_rec = pd.concat([CoH_DF_rec, pd.DataFrame({"Patient": [patient], "Cell": cell_type, "Marker": marker_dict[marker], "Mean": mean})])

    CoH_DF_rec.to_csv(join(path_here, "coh/data/CoH_Rec_DF.csv"))
    
    return CoH_DF_rec


def make_CoH_Tensor_rec():
    """Processes RA DataFrame into Xarray Tensor"""
    CoH_DF_rec = pd.read_csv(join(path_here, "coh/data/CoH_Rec_DF.csv"))
    patients = CoH_DF_rec.Patient.unique()
    cells = CoH_DF_rec.Cell.unique()
    markers = np.array(["IFNg R1", "TGFB RII", "PD1", "PD_L1", "IL2Ra", "IL2RB", "IL4Ra", "IL6Ra", "IL6RB", "IL7Ra", "IL10R", "IL12RI"])

    tensor = np.empty((len(patients), len(cells), len(markers)))
    tensor[:] = np.nan
    for i, pat in enumerate(patients):
        print(pat)
        for j, cell in enumerate(cells):
            for k, mark in enumerate(markers):
                entry = CoH_DF_rec.loc[(CoH_DF_rec.Patient == pat) & (CoH_DF_rec.Cell == cell) & (CoH_DF_rec.Marker == mark)]["Mean"].values
                tensor[i, j, k] = np.mean(entry)

    # Normalize
    for i, _ in enumerate(markers):
        tensor[:, :, i][~np.isnan(tensor[:, :, i])] -= np.mean(tensor[:, :, i])
        tensor[:, :, i][~np.isnan(tensor[:, :, i])] /= np.std(tensor[:, :, i])

    CoH_xarray = xa.DataArray(tensor, dims=("Patient", "Cell", "Marker"), coords={"Patient": patients, "Cell": cells, "Marker": markers})

    CoH_xarray.to_netcdf(join(path_here, "coh/data/CoH_Rec.nc"))
    return tensor
