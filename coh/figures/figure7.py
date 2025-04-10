"""This creates Figure 7, heatmap (clustered factor correlations)."""

import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage

from ..flow import make_CoH_Tensor
from ..flow_rec import make_CoH_Tensor_rec
from ..tensor import factorTensor
from .common import getSetup, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 1))

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    CoH_Data = make_CoH_Tensor(just_signal=True)

    with open("./coh/data/signaling.pkl", "rb") as ff:
        tFacAllM = pickle.load(ff)  # 12 component

    CoH_Data_R = make_CoH_Tensor_rec()
    tFacAllM_R = factorTensor(CoH_Data_R.to_numpy(), r=5)

    return CoH_Factor_HM(tFacAllM, CoH_Data, tFacAllM_R, CoH_Data_R)



def CoH_Factor_HM(tFac, CoH_Array, tFac_R, CoH_Array_R):
    """Plots bar plot for spec."""
    coord = CoH_Array.dims.index("Patient")
    mode_facs = tFac[1][coord]

    # BC_Patients = get_status_df().loc[status_DF.Status == "BC"].Patient.unique()

    tFacDF = pd.DataFrame(
        mode_facs,
        columns=["Sig. Comp " + str(i + 1) for i in range(mode_facs.shape[1])],
        index=CoH_Array["Patient"],
    )

    coord_R = CoH_Array_R.dims.index("Patient")
    mode_facs_R = tFac_R[1][coord_R]

    tFacDF_R = pd.DataFrame(
        mode_facs_R,
        columns=["Rec. Comp " + str(i + 1) for i in range(mode_facs_R.shape[1])],
        index=CoH_Array_R["Patient"],
    )

    plot_DF = tFacDF.join(tFacDF_R, how="inner")
    cDF = plot_DF.corr()

    Z = linkage(np.abs(cDF), optimal_ordering=True)

    cmap = sns.color_palette("vlag", as_cmap=True)
    return sns.clustermap(
        data=cDF,
        robust=True,
        vmin=-1,
        vmax=1,
        row_cluster=True,
        col_cluster=True,
        annot=True,
        cmap=cmap,
        figsize=(8, 8),
        row_linkage=Z,
        col_linkage=Z,
    )
