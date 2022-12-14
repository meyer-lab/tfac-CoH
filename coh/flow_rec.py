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
from FlowCytometryTools import PolyGate, FCMeasurement
from .flow import form_gate, pop_gate, live_PBMC_gate, pop_gate, process_sample, get_gate_dict

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


def compile_patient_rec(patient_num, cellFrac):
    """Adds all data from a single patient to an FC file"""
    pathname = "/opt/CoH/Receptor Data/F" + patient_num.split(" ")[1] + "_Unmixed.fcs"
    FCfiles = []
    FCfiles.append(FCMeasurement(ID=patient_num, datafile=pathname))
    return combineWells_rec(FCfiles, cellFrac, patient_num.split(" ")[1])


def combineWells_rec(samples, cellFrac, patientNum):
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


gate_dict = get_gate_dict()


def make_flow_df_rec(subtract=True, abundance=False, foldChange=False):
    """Compiles data for all populations for all patients into .csv"""
    patients = ["Patient 26", "Patient 28", "Patient 30", "Patient 34", "Patient 35", "Patient 43", "Patient 44", "Patient 45", "Patient 52", "Patient 52A", "Patient 54", "Patient 56", "Patient 58", "Patient 60", "Patient 61", "Patient 62", "Patient 63", "Patient 66", "Patient 68", "Patient 69", "Patient 70", "Patient 79", "Patient 4", "Patient 8", "Patient 406", "Patient 10-T1",  "Patient 10-T2",  "Patient 10-T3", "Patient 15-T1", "Patient 15-T2", "Patient 15-T3", "Patient 19186-2", "Patient 19186-3", "Patient 19186-14", "Patient 21368-3", "Patient 21368-4"]
    cell_types = ["T", "CD16 NK", "CD8+", "CD4+", "CD4-/CD8-", "Treg", "Treg 1", "Treg 2", "Treg 3", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 TEMRA",
                  "CD4 TEM", "CD4 TCM", "CD4 Naive", "CD4 TEMRA", "CD20 B", "CD20 B Naive", "CD20 B Memory", "CD33 Myeloid", "Classical Monocyte", "NC Monocyte"]
    gateDF = pd.read_csv(join(path_here, "coh/data/CoH_Flow_Gates_Receptors.csv")).reset_index().drop("Unnamed: 0", axis=1)
    CoH_DF = pd.DataFrame([])
    markerKey = pd.read_csv(join(path_here, "coh/data/Patient_Receptor_Panels.csv"))

    for patient in patients:
        patient_num = patient.split(" ")[1]
        sample = FCMeasurement(ID="Sample", datafile="/opt/CoH/Receptor Data/F" + patient_num + "_Unmixed.fcs")
        sample, markers = process_sample(sample)
        sample = live_PBMC_gate(sample, patient, gateDF)
        for cell_type in cell_types:
            pop_sample, _ = pop_gate(sample, cell_type, patient, gateDF)
            for marker in markers:
                dictionary = markerKey.loc[markerKey.Patient == patient].Panel.values
                marker_dict = panelDict[dictionary[0]]
                mean = pop_sample.data[marker_dict[marker]]
                mean = np.mean(mean.values[mean.values < np.quantile(mean.values, 0.995)])
                CoH_DF = pd.concat([CoH_DF, pd.DataFrame({"Patient": [patient], "Cell": cell_type, "Marker": marker_dict[marker], "Mean": mean})])

    CoH_DF.to_csv(join(path_here, "coh/data/CoH_Rec_DF.csv"))
    
    return CoH_DF


def make_CoH_Tensor_rec(subtract=True, just_signal=False, foldChange=False):
    """Processes RA DataFrame into Xarray Tensor"""
    if subtract:
        if foldChange:
            CoH_DF = pd.read_csv(join(path_here, "coh/data/CoH_Flow_DF_FC.csv"))
        else:
            CoH_DF = pd.read_csv(join(path_here, "coh/data/CoH_Flow_DF.csv"))
        CoH_DF.loc[(CoH_DF.Treatment == "Untreated"), "Mean"] = 0
    else:
        CoH_DF = pd.read_csv(join(path_here, "coh/data/NN_CoH_Flow_DF.csv"))
    patients = CoH_DF.Patient.unique()
    times = CoH_DF.Time.unique()
    treatments = CoH_DF.Treatment.unique()
    cells = CoH_DF.Cell.unique()
    if just_signal or foldChange:
        markers = np.array(["pSTAT1", "pSTAT3", "pSTAT4", "pSTAT5", "pSTAT6", "pSmad1-2"])
    else:
        markers = CoH_DF.Marker.unique()

    tensor = np.empty((len(patients), len(times), len(treatments), len(cells), len(markers)))
    tensor[:] = np.nan
    for i, pat in enumerate(patients):
        print(pat)
        for j, tim in enumerate(times):
            for k, treat in enumerate(treatments):
                for ii, cell in enumerate(cells):
                    for jj, mark in enumerate(markers):
                        entry = CoH_DF.loc[(CoH_DF.Patient == pat) & (CoH_DF.Time == tim) & (CoH_DF.Treatment == treat) & (CoH_DF.Cell == cell) & (CoH_DF.Marker == mark)]["Mean"].values
                        tensor[i, j, k, ii, jj] = np.mean(entry)

    # Normalize
    for i, _ in enumerate(markers):
        tensor[:, :, :, :, i][~np.isnan(tensor[:, :, :, :, i])] /= np.nanmax(tensor[:, :, :, :, i])

    CoH_xarray = xa.DataArray(tensor, dims=("Patient", "Time", "Treatment", "Cell", "Marker"), coords={"Patient": patients, "Time": times, "Treatment": treatments, "Cell": cells, "Marker": markers})
    if subtract:
        if foldChange:
            CoH_xarray.to_netcdf(join(path_here, "coh/data/CoH_Tensor_DataSet_FC.nc"))
        else:
            if just_signal:
                CoH_xarray.to_netcdf(join(path_here, "coh/data/CoHTensorDataJustSignal.nc"))
            else:
                CoH_xarray.to_netcdf(join(path_here, "coh/data/CoH Tensor DataSet.nc"))
    else:
        CoH_xarray.to_netcdf(join(path_here, "coh/data/NN CoH Tensor DataSet.nc"))
    return tensor


def make_CoH_Tensor_abund():
    """Processes RA DataFrame into Xarray Tensor"""
    CoH_DF = pd.read_csv(join(path_here, "coh/data/CoH_Flow_DF_Abund.csv"))

    patients = CoH_DF.Patient.unique()
    times = CoH_DF.Time.unique()
    treatments = CoH_DF.Treatment.unique()
    cells = CoH_DF.Cell.unique()

    tensor = np.empty((len(patients), len(times), len(treatments), len(cells)))
    tensor[:] = np.nan
    for i, pat in enumerate(patients):
        print(pat)
        for j, tim in enumerate(times):
            for k, treat in enumerate(treatments):
                for ii, cell in enumerate(cells):
                    entry = CoH_DF.loc[(CoH_DF.Patient == pat) & (CoH_DF.Time == tim) & (CoH_DF.Treatment == treat) & (CoH_DF.Cell == cell)]["Abundance"].values
                    tensor[i, j, k, ii] = np.mean(entry)

    # Normalize
    for i, _ in enumerate(cells):
        tensor[:, :, :, i][~np.isnan(tensor[:, :, :, i])] /= np.nanmax(tensor[:, :, :, i])
        tensor[:, :, :, i][~np.isnan(tensor[:, :, :, i])] -= np.nanmean(tensor[:, :, :, i])

    CoH_xarray = xa.DataArray(tensor, dims=("Patient", "Time", "Treatment", "Cell"), coords={"Patient": patients, "Time": times, "Treatment": treatments, "Cell": cells})
    CoH_xarray.to_netcdf(join(path_here, "coh/data/CoH_Tensor_Abundance.nc"))
    return tensor


def make_flow_sc_dataframe():
    """Compiles data for all populations for all patients into .nc"""
    # patients = ["Patient 26", "Patient 28", "Patient 30", "Patient 34", "Patient 35", "Patient 43", "Patient 44", "Patient 45", "Patient 52", 
    #             "Patient 52A", "Patient 54", "Patient 56", "Patient 58", "Patient 60", "Patient 61", "Patient 62", "Patient 63", "Patient 66", "Patient 68", 
    #             "Patient 69", "Patient 70", "Patient 79", "Patient 4", "Patient 8", "Patient 406", "Patient 10-T1",  "Patient 10-T2",  "Patient 10-T3", "Patient 15-T1", 
    #             "Patient 15-T2", "Patient 15-T3", "Patient 19186-2", "Patient 19186-3", "Patient 19186-14", "Patient 21368-3", "Patient 21368-4"]
    # times = ["15min", "60min"]
    patients = ['Patient 10-T1', 'Patient 10-T2', 'Patient 10-T3', 'Patient 15-T1',
        'Patient 15-T2', 'Patient 15-T3', 'Patient 19186-14', 'Patient 19186-2',
        'Patient 19186-3', 'Patient 21368-3', 'Patient 21368-4', 'Patient 26',
        'Patient 28', 'Patient 30', 'Patient 34', 'Patient 35', 'Patient 4',
        'Patient 406', 'Patient 43', 'Patient 44', 'Patient 45', 'Patient 52',
        'Patient 52A', 'Patient 54', 'Patient 56', 'Patient 58', 'Patient 60',
        'Patient 61', 'Patient 62', 'Patient 63', 'Patient 66', 'Patient 68',
        'Patient 69', 'Patient 70', 'Patient 79', 'Patient 8']
    times = ["15min"]
    # treatments = ["Untreated",
    #     "IFNg-1ng",
    #     "IFNg-1ng+IL6-1ng",
    #     "IFNg-1ng+IL6-50ng",
    #     "IFNg-50ng",
    #     "IFNg-50ng+IL6-1ng",
    #     "IFNg-50ng+IL6-50ng",
    #     "IL10-50ng",
    #     "IL12-100ng",
    #     "IL2-50ng",
    #     "IL4-50ng",
    #     "IL6-1ng",
    #     "IL6-50ng",
    #     "TGFB-50ng"]
    treatments = ['Untreated', 'IFNg-50ng', 'IL10-50ng', 'IL4-50ng', 'IL2-50ng', 'IL6-50ng']
    cell_types = ["T", "CD16 NK", "CD8+", "CD4+", "CD4-/CD8-", "Treg", "Treg 1", "Treg 2", "Treg 3", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 TEMRA",
                  "CD4 TEM", "CD4 TCM", "CD4 Naive", "CD4 TEMRA", "CD20 B", "CD20 B Naive", "CD20 B Memory", "CD33 Myeloid", "Classical Monocyte", "NC Monocyte"]
    gateDF = pd.read_csv(join(path_here, "coh/data/CoH_Flow_Gates.csv")).reset_index().drop("Unnamed: 0", axis=1)
    totalDF = pd.DataFrame([])

    for i, patient in enumerate(patients):
        patient_num = patient.split(" ")[1]
        patient_files = []
        for name in Path(r"" + str("/opt/CoH/" + patient + "/")).glob("**/*.fcs"):
            patient_files.append(str(name))
        for j, time in enumerate(times):
            for k, treatment in enumerate(treatments):
                print(patient, time, treatment)
                if ("/opt/CoH/" + patient + "/----F" + patient_num + "_" + time + "_" + treatment + "_Unmixed.fcs" in patient_files):
                    sample = FCMeasurement(ID="Sample", datafile="/opt/CoH/" + patient + "/----F" + patient_num + "_" + time + "_" + treatment + "_Unmixed.fcs")
                    sample, markers = process_sample(sample)
                    sample = live_PBMC_gate(sample, patient, gateDF)
                    for cell_type in cell_types:
                        pop_sample, abund = pop_gate(sample, cell_type, patient, gateDF)
                        CoH_DF = pop_sample.data.drop("Time", axis=1)
                        CoH_DF["Cell"] = np.arange(1, CoH_DF.shape[0] + 1)
                        CoH_DF["CellType"] = np.tile(cell_type, CoH_DF.shape[0])
                        CoH_DF["Time"] = np.tile(time, CoH_DF.shape[0])
                        CoH_DF["Treatment"] = np.tile(treatment, CoH_DF.shape[0])
                        CoH_DF["Patient"] = np.tile([patient], CoH_DF.shape[0])
                        totalDF = pd.concat([totalDF,CoH_DF])
                        print("CellType:",cell_type, "Shape:", CoH_DF.shape[0])

                    print(np.shape(totalDF))
                    
    totalDF.to_csv(join(path_here, "coh/data/CoH_Flow_SC.csv"))

    return totalDF