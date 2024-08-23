"""
This creates Figure 4's clustered heat map.
"""

import pandas as pd
import seaborn as sns
from .common import getSetup
from ..tensor import factorTensor
from ..flow_rec import make_CoH_Tensor_rec, get_status_rec_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    _, f = getSetup((8, 2), (1, 1))

    CoH_Data = make_CoH_Tensor_rec()
    tFacAllM = factorTensor(CoH_Data.values, r=5)
    f = plot_coh_clust(tFacAllM, CoH_Data, "Patient", get_status_rec_df())

    return f


def plot_coh_clust(tFac, CoH_Array, mode, status_df):
    """Plots tensor factorization of cells"""
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame(
        mode_facs.T,
        columns=mode_labels,
        index=[i + 1 for i in range(mode_facs.shape[1])],
    )

    cmap = sns.color_palette("vlag", as_cmap=True)

    status = status_df.reset_index()["Status"]
    lut = dict(zip(status.unique(), "rbg"))
    col_colors = pd.DataFrame(status.map(lut))
    col_colors["Patient"] = status_df.Patient.values
    col_colors = col_colors.set_index("Patient")
    f = sns.clustermap(
        data=tFacDF,
        robust=True,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        row_cluster=False,
        col_colors=col_colors,
        figsize=(8, 3),
    )
    return f
