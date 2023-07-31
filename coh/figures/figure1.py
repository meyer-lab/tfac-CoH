"""
This creates Figure 1.
"""
import numpy as np
import seaborn as sns
import tensorly as tl
from .common import subplotLabel, getSetup
from ..tensor import factorTensor
from ..flow import make_CoH_Tensor


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 3), multz={0: 2})

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    data = make_CoH_Tensor(just_signal=True)

    tFacAllM = factorTensor(data.values, r=12)
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

    fullHeatMap(ax[1], data.loc[:, :, :, "pSTAT3"], cbar=False)
    fullHeatMap(ax[2], data.loc[:, :, :, "pSTAT5"], cbar=False)
    fullHeatMap(ax[3], data.loc[:, :, :, "pSTAT6"], cbar=False)

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
