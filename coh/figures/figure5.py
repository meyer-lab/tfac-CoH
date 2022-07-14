"""
This creates Figure 1.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import cp_normalize, perform_CP
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..flow import make_flow_df, make_CoH_Tensor_abund

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    #make_flow_df(subtract=False, abundance=True)
    # make_CoH_Tensor_abund()

    num_comps = 6

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Tensor_Abundance.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    cp_normalize(tFacAllM)
    R2Xplot(ax[0], CoH_Data.values, compNum=10)
    plot_tFac_CoH(ax[1], tFacAllM, CoH_Data, "Patient", numComps=num_comps)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Time", numComps=num_comps)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Treatment", numComps=num_comps)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Cell", numComps=num_comps)


    return f


def factorTensor(tensor, numComps):
    """ Takes Tensor, and mask and returns tensor factorized form. """
    tfac = perform_CP(tensor, numComps, tol=1e-7, maxiter=1000)
    R2X = tfac.R2X
    tfac = cp_flip_sign(tfac)
    return tfac, R2X


def R2Xplot(ax, tensor, compNum):
    """Creates R2X plot for non-neg CP tensor decomposition"""
    varHold = np.zeros(compNum)
    for i in range(1, compNum + 1):
        print(i)
        _, R2X = factorTensor(tensor, i)
        varHold[i - 1] = R2X

    ax.scatter(np.arange(1, compNum + 1), varHold, c='k', s=20.)
    ax.set(title="R2X", ylabel="Variance Explained", xlabel="Number of Components", ylim=(0, 1), xlim=(0, compNum + 0.5), xticks=np.arange(0, compNum + 1))


def plot_tFac_CoH(ax, tFac, CoH_Array, mode, numComps=3):
    """Plots tensor factorization of cells"""
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame()

    for i in range(0, numComps):
        tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, i], "Component": (i + 1), mode: mode_labels})])

    tFacDF = pd.pivot(tFacDF, index="Component", columns=mode, values="Component_Val")
    cmap = sns.color_palette("vlag", as_cmap=True)
    if mode == "Patient":
        tFacDF = tFacDF[["Patient 35", "Patient 43", "Patient 44", "Patient 45", "Patient 52", "Patient 54", "Patient 56", "Patient 58", "Patient 63", "Patient 66", "Patient 70", "Patient 79", "Patient 4", "Patient 8", "Patient 406", "Patient 10-T1",  "Patient 10-T2",  "Patient 10-T3", "Patient 15-T1",  "Patient 15-T2",  "Patient 15-T3"]]
    sns.heatmap(data=tFacDF, ax=ax, cmap=cmap, vmin=-1, vmax=1)
