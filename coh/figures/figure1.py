"""
This creates Figure 1.
"""
import xarray as xa
import tensorly as tl
import numpy as np
import seaborn as sns
import pandas as pd
import os
from tensorly.decomposition import non_negative_parafac, parafac
from tensorpack.cmtf import cp_normalize
from .figureCommon import subplotLabel, getSetup
from os.path import join
from ..flow import make_flow_df, make_CoH_Tensor

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df()
    # make_CoH_Tensor(just_signal=True)

    num_comps = 10

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH Tensor DataSet Just Signal.nc"))
    tFacAllM = factorTensor(CoH_Data.values, numComps=num_comps)
    cp_normalize(tFacAllM)
    R2Xplot(ax[0], CoH_Data.values, compNum=25)
    plot_tFac_CoH(ax[1], tFacAllM, CoH_Data, "Patient", numComps=num_comps)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Time", numComps=num_comps)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Treatment", numComps=num_comps)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Cell", numComps=num_comps)
    plot_tFac_CoH(ax[5], tFacAllM, CoH_Data, "Marker", numComps=num_comps)


    return f


def factorTensor(tensor, numComps):
    """ Takes Tensor, and mask and returns tensor factorized form. """
    tfac = parafac(np.nan_to_num(tensor), rank=numComps, mask=np.isfinite(tensor), init='random', n_iter_max=5000, tol=1e-9, random_state=1)
    tensor = tensor.copy()
    tensor[np.isnan(tensor)] = tl.cp_to_tensor(tfac)[np.isnan(tensor)]
    return tfac


def R2Xplot(ax, tensor, compNum):
    """Creates R2X plot for non-neg CP tensor decomposition"""
    varHold = np.zeros(compNum)
    for i in range(1, compNum + 1):
        print(i)
        tFac = factorTensor(tensor, i)
        varHold[i - 1] = calcR2X(tensor, tFac)

    ax.scatter(np.arange(1, compNum + 1), varHold, c='k', s=20.)
    ax.set(title="R2X", ylabel="Variance Explained", xlabel="Number of Components", ylim=(0, 1), xlim=(0, compNum + 0.5), xticks=np.arange(0, compNum + 1))


def calcR2X(tensorIn, tensorFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - tensorIn)
    return 1.0 - tErr / np.nanvar(tensorIn)


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
        tFacDF = tFacDF[["Patient 35", "Patient 43", "Patient 44", "Patient 45", "Patient 52", "Patient 54", "Patient 56", "Patient 58", "Patient 63", "Patient 66", "Patient 70", "Patient 79", "Patient 4", "Patient 8", "Patient 406"]]
    sns.heatmap(data=tFacDF, ax=ax, cmap=cmap, vmin=-1, vmax=1)
