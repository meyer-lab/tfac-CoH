"""This file includes various methods for flow cytometry analysis of fixed cells."""

import ast
import textwrap
import warnings
from collections import OrderedDict
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xa
from FlowCytometryTools import FCMeasurement, PolyGate

warnings.filterwarnings("ignore")


def get_status_dict():
    """Returns status dictionary."""
    return OrderedDict(
        [
            ("Patient 26", "Healthy"),
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
            ("Patient 406", "BC"),
            ("Patient 19186-10-T1", "BC"),
            ("Patient 19186-10-T2", "BC"),
            ("Patient 19186-10-T3", "BC"),
            ("Patient 19186-15-T1", "BC"),
            ("Patient 19186-15-T2", "BC"),
            ("Patient 19186-15-T3", "BC"),
            ("Patient 19186-2", "BC"),
            ("Patient 19186-3", "BC"),
            ("Patient 19186-14", "BC"),
            ("Patient 21368-3", "BC"),
            ("Patient 21368-4", "BC"),
        ],
    )


def get_status_df():
    statusDF = pd.DataFrame.from_dict(get_status_dict(), orient="index").reset_index()
    statusDF.columns = ["Patient", "Status"]
    return statusDF


marker_dict = OrderedDict(
    {
        "Alexa Fluor 647-A": "pSTAT4",
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
        "BV510-A": "PD-L1",
    },
)


def compile_patient(patient_num, cellFrac):
    """Adds all data from a single patient to an FC file."""
    pathname = "/opt/CoH/" + str(patient_num) + "/"
    pathlist = Path(r"" + str(pathname)).glob("**/*.fcs")
    FCfiles = []
    for path in pathlist:
        FCfiles.append(FCMeasurement(ID=patient_num, datafile=path))
    return combineWells(FCfiles, cellFrac)


def combineWells(samples, cellFrac):
    """Accepts sample array returned from importF, and array of channels, returns combined well data."""
    markers = np.array(list(marker_dict.keys()))
    log_markers = markers[np.isin(markers, samples[0].data.columns)]
    samples[0] = samples[0].transform("tlog", channels=log_markers)
    combinedSamples = samples[0]
    for sample in samples[1:]:
        log_markers = markers[np.isin(markers, sample.data.columns)]
        sample = sample.transform("tlog", channels=log_markers)
        combinedSamples.data = pd.concat(
            [combinedSamples.data, sample.data.sample(frac=cellFrac)],
        )
    combinedSamples.data = combinedSamples.data.rename(marker_dict, axis=1)
    combinedSamples.data = combinedSamples.data.astype(int)
    return combinedSamples


def process_sample(sample):
    """Relabels and logs a sample."""
    markers = np.array(list(marker_dict.keys()))
    log_markers = markers[np.isin(markers, sample.data.columns)]
    sample = sample.transform("tlog", channels=log_markers)
    sample.data = sample.data.rename(marker_dict, axis=1)
    return sample, log_markers


def makeGate(lowerCorner, upperCorner, channels, name):
    """Returns square gate using upper and lower corners."""
    return PolyGate(
        [
            (lowerCorner[0], lowerCorner[1]),
            (upperCorner[0], lowerCorner[1]),
            (upperCorner[0], upperCorner[1]),
            (lowerCorner[0], upperCorner[1]),
        ],
        channels,
        region="in",
        name=name,
    )


gate_dict = OrderedDict(
    {
        "T": ["T Cell Gate"],
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
        "NC Monocyte": ["Non-Classical Monocyte Gate"],
    },
)


def get_gate_dict():
    return gate_dict


def form_gate(gate):
    """Deconvolutes string flow gate object."""
    vertices = ast.literal_eval(
        textwrap.dedent(str(gate).split("Vertices: ")[1].split("Channel")[0]),
    )
    channels = ast.literal_eval(
        textwrap.dedent(str(gate).split("Channel(s): ")[1].split("Name")[0]),
    )
    return PolyGate(vertices, channels)


def live_PBMC_gate(sample, patient, gateDF):
    """Returns singlet lymphocyte live PBMCs for a patient."""
    gates = ["PBMC Gate", "Single Cell Gate 1", "Single Cell Gate 2", "Live/Dead Gate"]
    for gate_name in gates:
        gate = form_gate(
            gateDF.loc[
                (gateDF["Patient"] == patient) & (gateDF["Gate Label"] == gate_name)
            ].Gate.values[0],
        )
        sample = sample.gate(gate)
    return sample


def pop_gate(sample, cell_type, patient, gateDF):
    """Extracts cell population sample."""
    gates = gate_dict[cell_type]

    pop_sample = copy(sample)
    for gate_name in gates:
        gate = form_gate(
            gateDF.loc[
                (gateDF["Patient"] == patient) & (gateDF["Gate Label"] == gate_name)
            ].Gate.values[0],
        )
        pop_sample = pop_sample.gate(gate)
    abund = pop_sample.counts / sample.counts
    return pop_sample, abund


def make_flow_df(subtract=True, abundance=False, foldChange=False):
    """Compiles data for all populations for all patients into .csv."""
    patients = get_status_dict().keys()
    times = ["15min", "60min"]
    treatments = [
        "Untreated",
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
        "TGFB-50ng",
    ]
    cell_types = list(gate_dict.keys())
    gateDF = (
        pd.read_csv("./coh/data/CoH_Flow_Gates.csv")
        .reset_index()
        .drop("Unnamed: 0", axis=1)
    )
    CoH_DF = pd.DataFrame([])

    for patient in patients:
        patient_num = patient.split(" ")[1]
        patient_files = []
        for name in Path(r"" + str("/opt/CoH/" + patient + "/")).glob("**/*.fcs"):
            patient_files.append(str(name))
        for time in times:
            for treatment in treatments:
                if treatment == "Untreated" and subtract:
                    untreatedDF = pd.DataFrame()
                if (
                    "/opt/CoH/"
                    + patient
                    + "/----F"
                    + patient_num
                    + "_"
                    + time
                    + "_"
                    + treatment
                    + "_Unmixed.fcs"
                    in patient_files
                ):
                    sample = FCMeasurement(
                        ID="Sample",
                        datafile="/opt/CoH/"
                        + patient
                        + "/----F"
                        + patient_num
                        + "_"
                        + time
                        + "_"
                        + treatment
                        + "_Unmixed.fcs",
                    )
                    sample, markers = process_sample(sample)
                    sample = live_PBMC_gate(sample, patient, gateDF)
                    for cell_type in cell_types:
                        pop_sample, abund = pop_gate(sample, cell_type, patient, gateDF)
                        if abundance:
                            CoH_DF = pd.concat(
                                [
                                    CoH_DF,
                                    pd.DataFrame(
                                        {
                                            "Patient": [patient],
                                            "Time": time,
                                            "Treatment": treatment,
                                            "Cell": cell_type,
                                            "Abundance": abund,
                                        },
                                    ),
                                ],
                            )
                        else:
                            for marker in markers:
                                mean = pop_sample.data[marker_dict[marker]]
                                mean = np.mean(
                                    mean.values[
                                        mean.values < np.quantile(mean.values, 0.995)
                                    ],
                                )
                                if subtract:
                                    if treatment == "Untreated":
                                        CoH_DF = pd.concat(
                                            [
                                                CoH_DF,
                                                pd.DataFrame(
                                                    {
                                                        "Patient": [patient],
                                                        "Time": time,
                                                        "Treatment": treatment,
                                                        "Cell": cell_type,
                                                        "Marker": marker_dict[marker],
                                                        "Mean": mean,
                                                    },
                                                ),
                                            ],
                                        )
                                        untreatedDF = pd.concat(
                                            [
                                                untreatedDF,
                                                pd.DataFrame(
                                                    {
                                                        "Cell": cell_type,
                                                        "Marker": marker_dict[marker],
                                                        "Mean": [mean],
                                                    },
                                                ),
                                            ],
                                        )
                                    else:
                                        subVal = untreatedDF.loc[
                                            (
                                                untreatedDF["Marker"]
                                                == marker_dict[marker]
                                            )
                                            & (untreatedDF["Cell"] == cell_type)
                                        ]["Mean"].values
                                        if foldChange:
                                            CoH_DF = pd.concat(
                                                [
                                                    CoH_DF,
                                                    pd.DataFrame(
                                                        {
                                                            "Patient": [patient],
                                                            "Time": time,
                                                            "Treatment": treatment,
                                                            "Cell": cell_type,
                                                            "Marker": marker_dict[
                                                                marker
                                                            ],
                                                            "Mean": (mean / subVal) - 1,
                                                        },
                                                    ),
                                                ],
                                            )
                                        else:
                                            CoH_DF = pd.concat(
                                                [
                                                    CoH_DF,
                                                    pd.DataFrame(
                                                        {
                                                            "Patient": [patient],
                                                            "Time": time,
                                                            "Treatment": treatment,
                                                            "Cell": cell_type,
                                                            "Marker": marker_dict[
                                                                marker
                                                            ],
                                                            "Mean": mean - subVal,
                                                        },
                                                    ),
                                                ],
                                            )
                                else:
                                    CoH_DF = pd.concat(
                                        [
                                            CoH_DF,
                                            pd.DataFrame(
                                                {
                                                    "Patient": [patient],
                                                    "Time": time,
                                                    "Treatment": treatment,
                                                    "Cell": cell_type,
                                                    "Marker": marker_dict[marker],
                                                    "Mean": mean,
                                                },
                                            ),
                                        ],
                                    )
                else:
                    for cell_type in cell_types:
                        for marker in markers:
                            if abundance:
                                CoH_DF = pd.concat(
                                    [
                                        CoH_DF,
                                        pd.DataFrame(
                                            {
                                                "Patient": [patient],
                                                "Time": time,
                                                "Treatment": treatment,
                                                "Cell": cell_type,
                                                "Abundance": np.nan,
                                            },
                                        ),
                                    ],
                                )
                            else:
                                CoH_DF = pd.concat(
                                    [
                                        CoH_DF,
                                        pd.DataFrame(
                                            {
                                                "Patient": [patient],
                                                "Time": time,
                                                "Treatment": treatment,
                                                "Cell": cell_type,
                                                "Marker": marker_dict[marker],
                                                "Mean": np.nan,
                                            },
                                        ),
                                    ],
                                )
    if subtract:
        UntreatedDF = CoH_DF.loc[(CoH_DF.Treatment == "Untreated")]
        UntreatedDF.to_csv("./coh/data/CoH_Flow_DF_Basal.csv")
        if foldChange:
            CoH_DF.loc[(CoH_DF.Treatment == "Untreated"), "Mean"] = 0
            CoH_DF.to_csv("./coh/data/CoH_Flow_DF_FC.csv")
        else:
            CoH_DF.to_csv("./coh/data/CoH_Flow_DF.csv")
    elif abundance:
        CoH_DF.to_csv("./coh/data/CoH_Flow_DF_Abund.csv")
    else:
        msg = "Shouldn't end up here."
        raise RuntimeError(msg)

    return CoH_DF


def make_CoH_Tensor(
    just_signal: bool = False, foldChange: bool = False,
) -> xa.DataArray:
    """Processes RA DataFrame into Xarray Tensor."""
    if foldChange:
        df = pd.read_csv("./coh/data/CoH_Flow_DF_FC.csv", index_col=[1, 2, 3, 4, 5])
    else:
        df = pd.read_csv("./coh/data/CoH_Flow_DF.csv", index_col=[1, 2, 3, 4, 5])

    xdata = df.to_xarray()["Mean"]
    xdata = xdata.loc[:, "15min", :, :, :]

    if just_signal or foldChange:
        markers = np.array(
            ["pSTAT1", "pSTAT3", "pSTAT4", "pSTAT5", "pSTAT6", "pSmad1-2"],
        )
        xdata = xdata.loc[:, :, :, markers]

    if foldChange:
        treatments = np.array(
            [
                "IL2-50ng",
                "IL4-50ng",
                "IL6-50ng",
                "IL10-50ng",
                "IFNg-50ng",
                "TGFB-50ng",
                "IFNg-50ng+IL6-50ng",
            ],
        )
    else:
        treatments = np.array(
            [
                "IL2-50ng",
                "IL4-50ng",
                "IL6-50ng",
                "IL10-50ng",
                "IFNg-50ng",
                "TGFB-50ng",
                "IFNg-50ng+IL6-50ng",
                "Untreated",
            ],
        )

    xdata = xdata.loc[list(get_status_dict().keys()), treatments, :, :]

    # Normalize
    if foldChange:
        xdata -= np.nanmean(xdata, axis=(0, 1, 2))[
            np.newaxis, np.newaxis, np.newaxis, :,
        ]
        xdata /= np.nanstd(xdata, axis=(0, 1, 2))[np.newaxis, np.newaxis, np.newaxis, :]
    else:
        xdata[:, :-1, :, :] -= np.nanmean(xdata[:, :-1, :, :], axis=(0, 1, 2))[
            np.newaxis, np.newaxis, np.newaxis, :,
        ]
        xdata[:, :-1, :, :] /= np.nanstd(xdata[:, :-1, :, :], axis=(0, 1, 2))[
            np.newaxis, np.newaxis, np.newaxis, :,
        ]
        xdata[:, -1, :, :] -= np.nanmean(xdata[:, -1, :, :], axis=(0, 1))[
            np.newaxis, np.newaxis, :,
        ]
        xdata[:, -1, :, :] /= np.nanstd(xdata[:, -1, :, :], axis=(0, 1))[
            np.newaxis, np.newaxis, :,
        ]

    return xdata


def make_CoH_Tensor_abund() -> xa.DataArray:
    """Processes RA DataFrame into Xarray Tensor."""
    df = pd.read_csv("./coh/data/CoH_Flow_DF_Abund.csv", index_col=[1, 2, 3, 4])
    df = df.dropna()

    xdata = df.to_xarray()["Abundance"]

    treatments = np.array(
        [
            "IL2-50ng",
            "IL4-50ng",
            "IL6-50ng",
            "IL10-50ng",
            "IFNg-50ng",
            "TGFB-50ng",
            "IFNg-50ng+IL6-50ng",
            "Untreated",
        ],
    )

    xdata = xdata.loc[list(get_status_dict().keys()), "15min", treatments, :]

    # Normalize
    xdata -= np.nanmean(xdata, axis=(0, 1))[np.newaxis, np.newaxis, :]
    xdata /= np.nanstd(xdata, axis=(0, 1))[np.newaxis, np.newaxis, :]

    return xdata


def make_flow_sc_dataframe():
    """Compiles data for all populations for all patients into a CSV."""
    patients = get_status_dict().keys()
    times = ["15min"]
    treatments = ["Untreated", "IL10-50ng"]
    cell_types = ["CD20 B", "CD20 B Naive", "CD20 B Memory", "CD8+"]
    gateDF = (
        pd.read_csv("./coh/data/CoH_Flow_Gates.csv")
        .reset_index()
        .drop("Unnamed: 0", axis=1)
    )
    totalDF = pd.DataFrame([])

    for _i, patient in enumerate(patients):
        patient_num = patient.split(" ")[1]
        patient_files = []
        for name in Path(r"" + str("/opt/CoH/" + patient + "/")).glob("**/*.fcs"):
            patient_files.append(str(name))
        for _j, time in enumerate(times):
            for _k, treatment in enumerate(treatments):
                if (
                    "/opt/CoH/"
                    + patient
                    + "/----F"
                    + patient_num
                    + "_"
                    + time
                    + "_"
                    + treatment
                    + "_Unmixed.fcs"
                    in patient_files
                ):
                    sample = FCMeasurement(
                        ID="Sample",
                        datafile="/opt/CoH/"
                        + patient
                        + "/----F"
                        + patient_num
                        + "_"
                        + time
                        + "_"
                        + treatment
                        + "_Unmixed.fcs",
                    )
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

    totalDF.to_csv("./coh/data/CoH_Flow_SC_IL10.csv")

    return totalDF
