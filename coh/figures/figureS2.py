"""
This creates Figure 1.
"""
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from ..flow import make_CoH_Tensor


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((3, 4), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    data = make_CoH_Tensor(just_signal=True)


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
    data.data[~np.isnan(data.data)] = 1
    data.data[np.isnan(data.data)] = 0

    fullHeatMap(ax[0], data.loc[:, :, :, "pSTAT1"], cbar=False)

    return f


def fullHeatMap(ax, data, cbar=True):
    """Plots the various affinities for IL-2 Muteins"""
    dataFlat = data.stack(condition=["Cell", "Patient"]).T
    dataFlat = dataFlat.to_pandas()
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
