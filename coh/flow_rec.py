"""
This file includes various methods for flow cytometry analysis of fixed cells.
"""
from collections import OrderedDict
import pandas as pd
import numpy as np
import warnings
import xarray as xa
from FlowCytometryTools import FCMeasurement
from .flow import pop_gate, live_PBMC_gate, pop_gate, get_gate_dict


warnings.filterwarnings("ignore")


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


def get_status_dict_rec():
    """Returns status dictionary"""
    return OrderedDict([("Patient 26", "Healthy"),
                        ("Patient 28", "Healthy"),
                        ("Patient 30", "Healthy"),
                        ("Patient 34", "Healthy"),
                        ("Patient 35", "Healthy"),
                        ("Patient 43", "Healthy"),
                        ("Patient 44", "Healthy"),
                        ("Patient 45", "Healthy"),
                        ("Patient 52", "Healthy"),
                        ("Patient 52A", "Healthy"),
                        ("Patient 54", "Healthy"),
                        ("Patient 56", "Healthy"),
                        ("Patient 58", "Healthy"),
                        ("Patient 60", "Healthy"),
                        ("Patient 61", "Healthy"),
                        ("Patient 62", "Healthy"),
                        ("Patient 63", "Healthy"),
                        ("Patient 66", "Healthy"),
                        ("Patient 68", "Healthy"),
                        ("Patient 69", "Healthy"),
                        ("Patient 70", "Healthy"),
                        ("Patient 79", "Healthy"),
                        ("Patient 19186-4", "BC"),
                        ("Patient 19186-8", "BC"),
                        ("Patient 19186-10-T1", "BC"),
                        ("Patient 19186-10-T2", "BC"),
                        ("Patient 19186-10-T3", "BC"),
                        ("Patient 19186-15-T1", "BC"),
                        ("Patient 19186-15-T2", "BC"),
                        ("Patient 19186-15-T3", "BC"),
                        ("Patient 19186-2", "BC"),
                        ("Patient 19186-3", "BC"),
                        ("Patient 19186-12", "BC"),
                        ("Patient 19186-14", "BC"),
                        ("Patient 21368-3", "BC"),
                        ("Patient 21368-4", "BC")])

def get_status_rec_df():
    statusDF = pd.DataFrame.from_dict(get_status_dict_rec(), orient='index').reset_index()
    statusDF.columns = ["Patient", "Status"]
    return statusDF


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


def make_flow_df_rec():
    """Compiles data for all populations for all patients into .csv"""
    patients = list(get_status_dict_rec().keys())
    cell_types = list(get_gate_dict().keys())
    gateDF = pd.read_csv("./coh/data/CoH_Flow_Gates_Receptors.csv").reset_index().drop("Unnamed: 0", axis=1)
    CoH_DF_rec = pd.DataFrame([])
    markerKey = pd.read_csv("./coh/data/Patient_Receptor_Panels.csv")

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

    CoH_DF_rec.to_csv("./coh/data/CoH_Rec_DF.csv")

    return CoH_DF_rec


def make_CoH_Tensor_rec() -> xa.DataArray:
    """Processes RA DataFrame into Xarray Tensor"""
    df = pd.read_csv("./coh/data/CoH_Rec_DF.csv", index_col=[1, 2, 3])

    xdata = df.to_xarray()["Mean"]
    markers = np.array(["IFNg R1", "TGFB RII", "PD1", "PD_L1", "IL2Ra", "IL2RB", "IL4Ra", "IL6Ra", "IL6RB", "IL7Ra", "IL10R", "IL12RI"])

    xdata = xdata.loc[:, :, markers]

    # Normalize
    xdata -= np.nanmean(xdata, axis=(0, 1))[np.newaxis, np.newaxis, :]
    xdata /= np.nanstd(xdata, axis=(0, 1))[np.newaxis, np.newaxis, :]

    return xdata


def make_flow_sc_dataframe_rec():
    """Compiles data for all populations for all patients into a CSV."""
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
        "Patient 19186-2",
        "Patient 19186-3",
        "Patient 19186-4",
        "Patient 19186-8",
        "Patient 19186-10-T1",
        "Patient 19186-10-T2",
        "Patient 19186-10-T3",
        "Patient 19186-15-T1",
        "Patient 19186-15-T2",
        "Patient 19186-15-T3",
        "Patient 19186-12",
        "Patient 19186-14",
        "Patient 21368-3",
        "Patient 21368-4"]
    cell_types = list(get_gate_dict().keys())
    gateDF = pd.read_csv("./coh/data/CoH_Flow_Gates_Receptors.csv").reset_index().drop("Unnamed: 0", axis=1)
    totalDF = pd.DataFrame([])
    markerKey = pd.read_csv("./coh/data/Patient_Receptor_Panels.csv")


    for i, patient in enumerate(patients):
        patient_num = patient.split(" ")[1]
        print(patient)
        dictionary = markerKey.loc[markerKey.Patient == patient_num].Panel.values
        marker_dict = panelDict[dictionary[0]]
        sample = FCMeasurement(ID="Sample", datafile="/opt/CoH/Receptor Data/F" + patient_num + "_Unmixed.fcs")
        sample, markers = process_sample_rec(sample, marker_dict)
        sample = live_PBMC_gate(sample, patient, gateDF)
        for cell_type in cell_types:
            pop_sample, abund = pop_gate(sample, cell_type, patient, gateDF)
            CoH_DF = pop_sample.data.drop("Time", axis=1)
            CoH_DF["Cell"] = np.arange(1, CoH_DF.shape[0] + 1)
            CoH_DF["CellType"] = np.tile(cell_type, CoH_DF.shape[0])
            CoH_DF["Patient"] = np.tile([patient], CoH_DF.shape[0])
            totalDF = pd.concat([totalDF, CoH_DF])

    totalDF.to_csv("./coh/data/CoH_Flow_SC_Rec.csv")

    return totalDF