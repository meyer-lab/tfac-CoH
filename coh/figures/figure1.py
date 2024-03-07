"""
This creates Figure 1.
"""
import pickle
import numpy as np
import seaborn as sns
import tensorly as tl
import pandas as pd
from .common import subplotLabel, getSetup
from ..flow import make_CoH_Tensor


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 7), (3, 6), multz={0: 5})

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    data = make_CoH_Tensor(just_signal=True)

    with open('./coh/data/signaling.pkl', 'rb') as ff:
        tFacAllM = pickle.load(ff) # 12 component

    tensor = tl.cp_to_tensor(tFacAllM)
    data.data[np.isnan(data.data)] = tensor[np.isnan(data.data)]

    treatments = [
        "IFNg-50ng",
        "IFNg-50ng+IL6-50ng",
        "IL10-50ng",
        "IL2-50ng",
        "IL4-50ng",
        "IL6-50ng",
        "TGFB-50ng",
        "Untreated",
    ]

    data = data.loc[:, treatments, :, :]

    fullHeatMap(ax[1], data.loc[:, :, :, "pSTAT1"], cbar=False)
    fullHeatMap(ax[2], data.loc[:, :, :, "pSTAT3"], cbar=False)
    fullHeatMap(ax[3], data.loc[:, :, :, "pSTAT4"], cbar=False)
    fullHeatMap(ax[4], data.loc[:, :, :, "pSTAT5"], cbar=False)
    fullHeatMap(ax[5], data.loc[:, :, :, "pSTAT6"], cbar=False)
    fullHeatMap(ax[6], data.loc[:, :, :, "pSmad1-2"], cbar=False)

    CoH_DF = pd.read_csv("./coh/data/CoH_Flow_DF.csv")
    CoH_DF = CoH_DF.loc[CoH_DF.Time == "15min"]
    cytok_marker_plot(CoH_DF, "IFNg-50ng", ["CD16 NK", "CD20 B", "CD4+", "CD8+", "CD33 Myeloid", "Treg"], "pSTAT1", ax[7])
    cytok_marker_plot(CoH_DF, "IL10-50ng", ["CD16 NK", "CD20 B", "CD4+", "CD8+", "CD33 Myeloid", "Treg"], "pSTAT3", ax[8])
    multi_cytoks_plot(CoH_DF, ["IFNg-50ng", "IFNg-50ng+IL6-50ng", "IL10-50ng", "IL2-50ng", "IL4-50ng", "IL6-50ng", "TGFB-50ng", "Untreated"], ["CD33 Myeloid"], "pSTAT4", ax[9])
    cytok_marker_plot(CoH_DF, "IL2-50ng", ["CD16 NK", "CD20 B", "CD4+", "CD8+", "CD33 Myeloid", "Treg"], "pSTAT5", ax[10])
    cytok_marker_plot(CoH_DF, "IL4-50ng", ["CD16 NK", "CD20 B", "CD4+", "CD8+", "CD33 Myeloid", "Treg"], "pSTAT6", ax[11])
    multi_cytoks_plot(CoH_DF, ["TGFB-50ng", "Untreated"], ["CD16 NK", "CD20 B", "CD4+", "CD8+", "CD33 Myeloid", "Treg"], "pSmad1-2", ax[12])

    ax[7].set(ylim=(-300, 1200))
    ax[8].set(ylim=(0, 1500))
    ax[9].set(ylim=(2500, 6000))
    ax[10].set(ylim=(-500, 1000))
    ax[11].set(ylim=(0, 1200))
    ax[12].set(ylim=(0, 6000))


    return f


def fullHeatMap(ax, data, cbar=True):
    """Plots the various affinities for IL-2 Muteins"""
    dataFlat = data.stack(condition=["Cell", "Patient"]).T
    dataFlat = dataFlat.to_pandas()
    # dataFlat.iloc[:, :] = dataFlat.values - np.array(dataFlat.loc[:, "Basal"])[:, np.newaxis]
    dataFlat.iloc[:, :] = dataFlat.values / np.max(dataFlat.values)

    sns.heatmap(
        data=dataFlat,
        vmin=0,
        vmax=1,
        ax=ax,
        yticklabels=False,
        cbar=cbar,
        cbar_kws={"label": data.coords["Marker"]},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


def cytok_stim_plot(CoH_DF, cytok, cells, ax):
    """Plots cells responses across signaling products for a single stimulatiom"""
    CoH_DF = CoH_DF.loc[(CoH_DF.Treatment == cytok) & (CoH_DF.Cell.isin(cells)) & CoH_DF.Marker.isin(["pSTAT1", "pSTAT3", "pSTAT3", "pSTAT5", "pSTAT6", "pSmad1-2"])]
    sns.boxplot(data=CoH_DF, x="Cell", y="Mean", hue="Marker", palette='husl', showfliers=False, ax=ax)
    ax.set(ylabel="Response to " + cytok)


def cytok_marker_plot(CoH_DF, cytok, cells, marker, ax):
    """Plots cells responses across signaling products for a single stimulatiom"""
    CoH_DF = CoH_DF.loc[(CoH_DF.Treatment == cytok) & (CoH_DF.Cell.isin(cells)) & (CoH_DF.Marker == marker)]
    sns.boxplot(data=CoH_DF, x="Cell", y="Mean", palette='husl', showfliers=False, ax=ax)
    ax.set(ylabel=marker + " response to " + cytok)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)


def multi_cytoks_plot(CoH_DF, cytoks, cells, marker, ax):
    """Plots cells responses across signaling products for a single stimulatiom"""
    CoH_DF = CoH_DF.loc[(CoH_DF.Treatment.isin(cytoks)) & (CoH_DF.Cell.isin(cells)) & (CoH_DF.Marker == marker)]
    for patient in CoH_DF.Patient.unique():
        CoH_DF.loc[(CoH_DF.Treatment != "Untreated") & (CoH_DF.Patient == patient), "Mean"] += CoH_DF.loc[(CoH_DF.Treatment == "Untreated") & (CoH_DF.Patient == patient)].Mean.values
    sns.boxplot(data=CoH_DF, x="Cell", y="Mean", hue="Treatment", palette='husl', showfliers=False, ax=ax)
    ax.set(ylabel=marker + " Response")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
