"""
This creates Figure 3's clustered heat map.
"""
import xarray as xa
import pandas as pd
import seaborn as sns
from .figureCommon import getSetup
from os.path import join, dirname
from ..tensor import factorTensor
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 2), (1, 1))

    # Add subplot labels
    # subplotLabel(ax)
    # make_CoH_Tensor(just_signal=True, basal=False)

    num_comps = 10
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    f = plot_coh_clust(tFacAllM, CoH_Data, "Patient", numComps=num_comps)
    

    return f


def plot_coh_clust(tFac, CoH_Array, mode, numComps=12):
    """Plots tensor factorization of cells"""
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame()

    for i in range(0, numComps):
        tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, i], "Component": (i + 1), mode: mode_labels})])

    tFacDF = pd.pivot(tFacDF, index="Component", columns=mode, values="Component_Val")
    cmap = sns.color_palette("vlag", as_cmap=True)
    status_df = pd.read_csv(join(path_here, "data/Patient_Status.csv")).sort_values(by="Patient").reset_index()
    status = status_df["Status"]
    lut = dict(zip(status.unique(), "rbg"))
    col_colors = pd.DataFrame(status.map(lut))
    col_colors["Patient"] = status_df.Patient.values
    col_colors = col_colors.set_index("Patient")
    f = sns.clustermap(data=tFacDF, robust=True, cmap=cmap, vmin=-1, vmax=1, row_cluster=False, col_colors=col_colors, figsize=(8, 3))
    return f
