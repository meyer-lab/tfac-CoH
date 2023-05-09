"""
This file includes various methods for flow cytometry analysis of fixed cells.
"""
from os.path import dirname, join
from pathlib import Path
import ast
import textwrap
import pandas as pd
import numpy as np
import warnings
import xarray as xa
import itertools
from copy import copy
from FlowCytometryTools import PolyGate, FCMeasurement
from .tensor import get_status_dict

path_here = dirname(dirname(__file__))


warnings.filterwarnings("ignore")
gate_df = pd.DataFrame()


marker_dict = {"Alexa Fluor 647-A": "pSTAT4",
               "Alexa Fluor 700-A": "CD20",
               "BV650-A": "CD14",
               "APC-Cy7-A": "CD14",
               "V450-A": "pSTAT6",
               "BV786-A": "CD27",
               "BV570-A": "CD3",
               "BV750-A": "CD33",
               "BUV395-A": "CD45RA",
               "LIVE DEAD Blue-A": "Live/Dead",
               "BUV563-A": "CD4",
               "BUV737-A": "CD16",
               "BUV805-A": "CD8",
               "Alexa Fluor 488-A": "pSTAT3",
               "Brilliant Blue 515-A": "pSTAT3",
               "PerCP-Cy5.5-A": "pSTAT1",
               "PE-A": "pSmad1-2",
               "PE-CF594-A": "FoxP3",
               "PE-Cy7-A": "pSTAT5",
               "BV605-A": "PD-1",
               "BV510-A": "PD-L1"}


def compile_patient(patient_num, cellFrac):
    """Adds all data from a single patient to an FC file"""
    pathname = "/opt/CoH/" + str(patient_num) + "/"
    pathlist = Path(r"" + str(pathname)).glob("**/*.fcs")
    FCfiles = []
    for path in pathlist:
        FCfiles.append(FCMeasurement(ID=patient_num, datafile=path))
    return combineWells(FCfiles, cellFrac)


def combineWells(samples, cellFrac):
    """Accepts sample array returned from importF, and array of channels, returns combined well data"""
    markers = np.array(["Alexa Fluor 647-A", "Alexa Fluor 700-A", "BV650-A", "APC-Cy7-A", "V450-A", "BV786-A", "BV570-A", "BV750-A", "BUV395-A", "LIVE DEAD Blue-A",
                       "BUV563-A", "BUV737-A", "BUV805-A", "Alexa Fluor 488-A", "Brilliant Blue 515-A", "PerCP-Cy5.5-A", "PE-A", "PE-CF594-A", "PE-Cy7-A", "BV605-A", "BV510-A"])
    log_markers = markers[np.isin(markers, samples[0].data.columns)]
    samples[0] = samples[0].transform("tlog", channels=log_markers)
    combinedSamples = samples[0]
    for sample in samples[1:]:
        log_markers = markers[np.isin(markers, sample.data.columns)]
        sample = sample.transform("tlog", channels=log_markers)
        combinedSamples.data = pd.concat([combinedSamples.data, sample.data.sample(frac=cellFrac)])
    combinedSamples.data = combinedSamples.data.rename(marker_dict, axis=1)
    combinedSamples.data = combinedSamples.data.astype(int)
    return combinedSamples


def process_sample(sample):
    """Relabels and logs a sample"""
    markers = np.array(["Alexa Fluor 647-A", "Alexa Fluor 700-A", "BV650-A", "APC-Cy7-A", "V450-A", "BV786-A", "BV570-A", "BV750-A", "BUV395-A", "LIVE DEAD Blue-A",
                       "BUV563-A", "BUV737-A", "BUV805-A", "Alexa Fluor 488-A", "Brilliant Blue 515-A", "PerCP-Cy5.5-A", "PE-A", "PE-CF594-A", "PE-Cy7-A", "BV605-A", "BV510-A"])
    log_markers = markers[np.isin(markers, sample.data.columns)]
    sample = sample.transform("tlog", channels=log_markers)
    sample.data = sample.data.rename(marker_dict, axis=1)
    return sample, log_markers


def makeGate(lowerCorner, upperCorner, channels, name):
    """Returns square gate using upper and lower corners"""
    return PolyGate([(lowerCorner[0], lowerCorner[1]), (upperCorner[0], lowerCorner[1]), (upperCorner[0], upperCorner[1]), (lowerCorner[0], upperCorner[1])], channels, region='in', name=name)


gate_dict = {"T": ["T Cell Gate"],
             "CD16 NK": ["CD16 NK Gate"],
             "CD8+": ["T Cell Gate", "CD8 Gate"],
             "CD4+": ["T Cell Gate", "CD4 Gate"],
             "CD4-/CD8-": ["T Cell Gate", "CD4/CD8- Gate"],
             "Treg": ["T Cell Gate", "CD4 Gate", "Treg Gate"],
             "Treg 1": ["T Cell Gate", "CD4 Gate", "Treg Gate", "Treg1 Gate"],
             "Treg 2": ["T Cell Gate", "CD4 Gate", "Treg Gate", "Treg2 Gate"],
             "Treg 3": ["T Cell Gate", "CD4 Gate", "Treg Gate", "Treg3 Gate"],
             "CD8 TEM": ["T Cell Gate", "CD8 Gate", "CD8 TEM Gate"],
             "CD8 TCM": ["T Cell Gate", "CD8 Gate", "CD8 TCM Gate"],
             "CD8 Naive": ["T Cell Gate", "CD8 Gate", "CD8 Naive Gate"],
             "CD8 TEMRA": ["T Cell Gate", "CD8 Gate", "CD8 TEMRA Gate"],
             "CD4 TEM": ["T Cell Gate", "CD4 Gate", "CD4 TEM Gate"],
             "CD4 TCM": ["T Cell Gate", "CD4 Gate", "CD4 TCM Gate"],
             "CD4 Naive": ["T Cell Gate", "CD4 Gate", "CD4 Naive Gate"],
             "CD4 TEMRA": ["T Cell Gate", "CD4 Gate", "CD4 TEMRA Gate"],
             "CD20 B": ["CD20 B Gate"],
             "CD20 B Naive": ["CD20 B Gate", "B Naive Gate"],
             "CD20 B Memory": ["CD20 B Gate", "B Memory Gate"],
             "CD33 Myeloid": ["CD33 Myeloid Gate"],
             "Classical Monocyte": ["Classical Monocyte Gate"],
             "NC Monocyte": ["Non-Classical Monocyte Gate"]}


def get_gate_dict():
    return gate_dict


def form_gate(gate):
    """Deconvolutes string flow gate object"""
    vertices = ast.literal_eval(textwrap.dedent(str(gate).split("Vertices: ")[1].split("Channel")[0]))
    channels = ast.literal_eval(textwrap.dedent(str(gate).split("Channel(s): ")[1].split("Name")[0]))
    return PolyGate(vertices, channels)


def live_PBMC_gate(sample, patient, gateDF):
    """Returns singlet lymphocyte live PBMCs for a patient"""
    gates = ["PBMC Gate", "Single Cell Gate 1", "Single Cell Gate 2", "Live/Dead Gate"]
    for gate_name in gates:
        gate = form_gate(gateDF.loc[(gateDF["Patient"] == patient) & (gateDF["Gate Label"] == gate_name)].Gate.values[0])
        sample = sample.gate(gate)
    return sample


def pop_gate(sample, cell_type, patient, gateDF):
    """Extracts cell population sample"""
    gates = gate_dict[cell_type]

    pop_sample = copy(sample)
    for gate_name in gates:
        gate = form_gate(gateDF.loc[(gateDF["Patient"] == patient) & (gateDF["Gate Label"] == gate_name)].Gate.values[0])
        pop_sample = pop_sample.gate(gate)
    abund = pop_sample.counts / sample.counts
    return pop_sample, abund


def make_flow_df(subtract=True, abundance=False, foldChange=False):
    """Compiles data for all populations for all patients into .csv"""
    patients = get_status_dict().keys()
    times = ["15min", "60min"]
    treatments = ["Untreated",
                  "IFNg-1ng",
                  "IFNg-1ng+IL6-1ng",
                  "IFNg-1ng+IL6-50ng",
                  "IFNg-50ng",
                  "IFNg-50ng+IL6-1ng",
                  "IFNg-50ng+IL6-50ng",
                  "IL10-50ng",
                  "IL12-100ng",
                  "IL2-50ng",
                  "IL4-50ng",
                  "IL6-1ng",
                  "IL6-50ng",
                  "TGFB-50ng"]
    cell_types = ["T", "CD16 NK", "CD8+", "CD4+", "CD4-/CD8-", "Treg", "Treg 1", "Treg 2", "Treg 3", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 TEMRA",
                  "CD4 TEM", "CD4 TCM", "CD4 Naive", "CD4 TEMRA", "CD20 B", "CD20 B Naive", "CD20 B Memory", "CD33 Myeloid", "Classical Monocyte", "NC Monocyte"]
    gateDF = pd.read_csv(join(path_here, "coh/data/CoH_Flow_Gates.csv")).reset_index().drop("Unnamed: 0", axis=1)
    CoH_DF = pd.DataFrame([])

    for patient in patients:
        patient_num = patient.split(" ")[1]
        patient_files = []
        for name in Path(r"" + str("/opt/CoH/" + patient + "/")).glob("**/*.fcs"):
            patient_files.append(str(name))
        for time in times:
            for treatment in treatments:
                print(patient, time, treatment)
                if treatment == "Untreated" and subtract:
                    untreatedDF = pd.DataFrame()
                if ("/opt/CoH/" + patient + "/----F" + patient_num + "_" + time + "_" + treatment + "_Unmixed.fcs" in patient_files):
                    sample = FCMeasurement(ID="Sample", datafile="/opt/CoH/" + patient + "/----F" + patient_num + "_" + time + "_" + treatment + "_Unmixed.fcs")
                    sample, markers = process_sample(sample)
                    sample = live_PBMC_gate(sample, patient, gateDF)
                    for cell_type in cell_types:
                        pop_sample, abund = pop_gate(sample, cell_type, patient, gateDF)
                        if abundance:
                            CoH_DF = pd.concat([CoH_DF, pd.DataFrame({"Patient": [patient], "Time": time, "Treatment": treatment, "Cell": cell_type, "Abundance": abund})])
                        else:
                            for marker in markers:
                                mean = pop_sample.data[marker_dict[marker]]
                                mean = np.mean(mean.values[mean.values < np.quantile(mean.values, 0.995)])
                                if subtract:
                                    if treatment == "Untreated":
                                        CoH_DF = pd.concat([CoH_DF, pd.DataFrame({"Patient": [patient], "Time": time, "Treatment": treatment,
                                                           "Cell": cell_type, "Marker": marker_dict[marker], "Mean": mean})])
                                        untreatedDF = pd.concat([untreatedDF, pd.DataFrame({"Cell": cell_type, "Marker": marker_dict[marker], "Mean": [mean]})])
                                    else:
                                        subVal = untreatedDF.loc[(untreatedDF["Marker"] == marker_dict[marker]) & (untreatedDF["Cell"] == cell_type)]["Mean"].values
                                        if foldChange:
                                            CoH_DF = pd.concat([CoH_DF, pd.DataFrame({"Patient": [patient], "Time": time, "Treatment": treatment,
                                                               "Cell": cell_type, "Marker": marker_dict[marker], "Mean": (mean / subVal) - 1})])
                                        else:
                                            CoH_DF = pd.concat([CoH_DF, pd.DataFrame({"Patient": [patient], "Time": time, "Treatment": treatment,
                                                               "Cell": cell_type, "Marker": marker_dict[marker], "Mean": mean - subVal})])
                                else:
                                    CoH_DF = pd.concat([CoH_DF, pd.DataFrame({"Patient": [patient], "Time": time, "Treatment": treatment,
                                                       "Cell": cell_type, "Marker": marker_dict[marker], "Mean": mean})])
                else:
                    print("Skipped")
                    for cell_type in cell_types:
                        for marker in markers:
                            if abundance:
                                CoH_DF = pd.concat([CoH_DF, pd.DataFrame({"Patient": [patient], "Time": time, "Treatment": treatment, "Cell": cell_type, "Abundance": np.nan})])
                            else:
                                CoH_DF = pd.concat([CoH_DF, pd.DataFrame({"Patient": [patient], "Time": time, "Treatment": treatment,
                                                   "Cell": cell_type, "Marker": marker_dict[marker], "Mean": np.nan})])
    if subtract:
        UntreatedDF = CoH_DF.loc[(CoH_DF.Treatment == "Untreated")]
        UntreatedDF.to_csv(join(path_here, "coh/data/CoH_Flow_DF_Basal.csv"))
        if foldChange:
            CoH_DF.loc[(CoH_DF.Treatment == "Untreated"), "Mean"] = 0
            CoH_DF.to_csv(join(path_here, "coh/data/CoH_Flow_DF_FC.csv"))
        else:
            CoH_DF.to_csv(join(path_here, "coh/data/CoH_Flow_DF.csv"))
    else:
        if abundance:
            CoH_DF.to_csv(join(path_here, "coh/data/CoH_Flow_DF_Abund.csv"))
        else:
            CoH_DF.to_csv(join(path_here, "coh/data/NN_CoH_Flow_DF.csv"))

    return CoH_DF


def make_CoH_Tensor(just_signal=False, foldChange=False):
    """Processes RA DataFrame into Xarray Tensor"""
    if foldChange:
        CoH_DF = pd.read_csv(join(path_here, "coh/data/CoH_Flow_DF_FC.csv"), index_col=0)
    else:
        CoH_DF = pd.read_csv(join(path_here, "coh/data/CoH_Flow_DF.csv"), index_col=0)

    #CoH_DF = CoH_DF.loc[CoH_DF.Time == "15min"]
    CoH_DF.Treatment = CoH_DF.Treatment.replace(["Untreated"], "Basal")

    if just_signal or foldChange:
        markers = np.array(["pSTAT1", "pSTAT3", "pSTAT4", "pSTAT5", "pSTAT6", "pSmad1-2"])
        CoH_DF = CoH_DF.loc[CoH_DF.Marker.isin(markers)]

    if foldChange:
        treatments = np.array(["IL2-50ng", "IL4-50ng", "IL6-50ng", "IL10-50ng", "IFNg-50ng", "TGFB-50ng", "IFNg-50ng+IL6-50ng"])
    else:
        treatments = CoH_DF.Treatment.unique()#np.array(["IL2-50ng", "IL4-50ng", "IL6-50ng", "IL10-50ng", "IFNg-50ng", "TGFB-50ng", "IFNg-50ng+IL6-50ng", "Basal"])
    CoH_DF = CoH_DF.loc[CoH_DF.Treatment.isin(treatments)]

    CoH_DF = CoH_DF.groupby(["Patient", "Time", "Treatment", "Cell", "Marker"], sort=False).Mean.mean().reset_index()
    patients = CoH_DF.Patient.unique()
    times = ["15min", "60min"]
    treatments = CoH_DF.Treatment.unique()
    cells = CoH_DF.Cell.unique()
    markers = CoH_DF.Marker.unique()

    tensor = np.empty((len(patients), len(treatments), len(cells), len(markers)))
    tensor[:] = np.nan
    values_vec = CoH_DF.Mean.values

    for i, combination in enumerate(itertools.product(patients, times, treatments, cells, markers)):
        coords = [
            np.where(
                patients == combination[0])[0][0], np.where(
                times == combination[1])[0], np.where(
                treatments == combination[2])[0][0], np.where(
                    cells == combination[3])[0][0], np.where(
                        markers == combination[4])[0][0]]
        tensor[coords[0], coords[2], coords[3], coords[4]] = values_vec[i]

    # Normalize
    for i, _ in enumerate(markers):
        if foldChange:
            tensor[:, :, :, i][~np.isnan(tensor[:, :, :, i])] -= np.nanmean(tensor[:, :, :, i])
            tensor[:, :, :, i][~np.isnan(tensor[:, :, :, i])] /= np.nanstd(tensor[:, :, :, i])
        else:
            tensor[:, 1::, :, i][~np.isnan(tensor[:, 1::, :, i])] -= np.nanmean(tensor[:, 1::, :, i])
            tensor[:, 1::, :, i][~np.isnan(tensor[:, 1::, :, i])] /= np.nanstd(tensor[:, 1::, :, i])
            tensor[:, 0, :, i][~np.isnan(tensor[:, 0, :, i])] -= np.nanmean(tensor[:, 0, :, i])
            tensor[:, 0, :, i][~np.isnan(tensor[:, 0, :, i])] /= np.nanstd(tensor[:, 0, :, i])  # Basal Separate

    CoH_xarray = xa.DataArray(tensor, dims=("Patient", "Treatment", "Cell", "Marker"), coords={"Patient": patients, "Treatment": treatments, "Cell": cells, "Marker": markers})
    CoH_xarray = CoH_xarray.reindex(Marker=np.sort(markers))
    CoH_xarray = CoH_xarray.reindex(Cell=np.sort(cells))
    CoH_xarray = CoH_xarray.reindex(Treatment=np.sort(treatments))

    if foldChange:
        CoH_xarray.to_netcdf(join(path_here, "coh/data/CoH_Tensor_DataSet_FC.nc"))
    else:
        CoH_xarray.to_netcdf(join(path_here, "coh/data/CoH_Tensor_DataSet_All.nc"))

    return tensor


def make_CoH_Tensor_abund():
    """Processes RA DataFrame into Xarray Tensor"""
    CoH_DF = pd.read_csv(join(path_here, "coh/data/CoH_Flow_DF_Abund.csv"))

    treatments = np.array(["IL2-50ng", "IL4-50ng", "IL6-50ng", "IL10-50ng", "IFNg-50ng", "TGFB-50ng", "IFNg-50ng+IL6-50ng", "Untreated"])
    CoH_DF = CoH_DF.loc[CoH_DF.Treatment.isin(treatments)]
    patients = CoH_DF.Patient.unique()
    treatments = CoH_DF.Treatment.unique()
    cells = CoH_DF.Cell.unique()

    tensor = np.empty((len(patients), len(treatments), len(cells)))
    tensor[:] = np.nan
    for i, pat in enumerate(patients):
        print(pat)
        for k, treat in enumerate(treatments):
            for ii, cell in enumerate(cells):
                entry = CoH_DF.loc[(CoH_DF.Patient == pat) & (CoH_DF.Treatment == treat) & (CoH_DF.Cell == cell) & (CoH_DF.Time == "15min")]["Abundance"].values
                tensor[i, k, ii] = np.mean(entry)

    # Normalize
    for i, _ in enumerate(cells):
        tensor[:, :, i][~np.isnan(tensor[:, :, i])] -= np.nanmean(tensor[:, :, i])
        tensor[:, :, i][~np.isnan(tensor[:, :, i])] /= np.nanstd(tensor[:, :, i])

    CoH_xarray = xa.DataArray(tensor, dims=("Patient", "Treatment", "Cell"), coords={"Patient": patients, "Treatment": treatments, "Cell": cells})
    CoH_xarray.to_netcdf(join(path_here, "coh/data/CoH_Tensor_Abundance.nc"))
    return tensor


def make_flow_sc_dataframe():
    """Compiles data for all populations for all patients into .nc"""
    patients = get_status_dict().keys()
    times = ["15min"]
    treatments = ['Untreated', 'IFNg-50ng', 'IL10-50ng', 'IL4-50ng', 'IL2-50ng', 'IL6-50ng']
    cell_types = ["T", "CD16 NK", "CD8+", "CD4+", "CD4-/CD8-", "Treg", "Treg 1", "Treg 2", "Treg 3", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 TEMRA",
                  "CD4 TEM", "CD4 TCM", "CD4 Naive", "CD4 TEMRA", "CD20 B", "CD20 B Naive", "CD20 B Memory", "CD33 Myeloid", "Classical Monocyte", "NC Monocyte"]
    treatments = ['Untreated', 'IL10-50ng']
    cell_types = ["CD20 B", "CD20 B Naive", "CD20 B Memory", "CD8+"]
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
                        totalDF = pd.concat([totalDF, CoH_DF])

    totalDF.to_csv(join(path_here, "coh/data/CoH_Flow_SC_IL10.csv"))

    return totalDF
