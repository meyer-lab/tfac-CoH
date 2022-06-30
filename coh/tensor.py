import numpy as np
import seaborn as sns
import pandas as pd
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import cp_normalize, perform_CP


def factorTensor(tensor, numComps):
    """ Takes Tensor, and mask and returns tensor factorized form. """
    tfac = perform_CP(tensor, numComps, tol=1e-7, maxiter=1000)
    R2X = tfac.R2X
    tfac = cp_flip_sign(tfac)
    tfac = cp_normalize(tfac)
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


def plot_tFac_CoH(ax, tFac, CoH_Array, mode, numComps=3, nn=False):
    """Plots tensor factorization of cells"""
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame()

    for i in range(0, numComps):
        tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, i], "Component": (i + 1), mode: mode_labels})])

    tFacDF = pd.pivot(tFacDF, index="Component", columns=mode, values="Component_Val")
    if nn:
        sns.heatmap(data=tFacDF, ax=ax, vmin=0, vmax=1)
    else:
        cmap = sns.color_palette("vlag", as_cmap=True)
        sns.heatmap(data=tFacDF, ax=ax, cmap=cmap, vmin=-1, vmax=1)
