import tensorly as tl
import numpy as np
import seaborn as sns
import pandas as pd
from tensorly.decomposition import non_negative_parafac, parafac
from sklearn.linear_model import LogisticRegression


def factorTensor(tensor, numComps, nn=False):
    """ Takes Tensor, and mask and returns tensor factorized form. """
    if nn:
        tfac = non_negative_parafac(np.nan_to_num(tensor), rank=numComps, mask=np.isfinite(tensor), init='random', n_iter_max=5000, tol=1e-9, random_state=1)
    else:
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


def plot_tFac_CoH(ax, tFac, CoH_Array, mode, numComps=3, nn=False):
    """Plots tensor factorization of cells"""
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame()

    for i in range(0, numComps):
        tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, i], "Component": (i + 1), mode: mode_labels})])

    tFacDF = pd.pivot(tFacDF, index="Component", columns=mode, values="Component_Val")
    if mode == "Patient":
        tFacDF = tFacDF[["Patient 35", "Patient 43", "Patient 44", "Patient 45", "Patient 52", "Patient 54", "Patient 56", "Patient 58", "Patient 63", "Patient 66", "Patient 70", "Patient 79", "Patient 4", "Patient 8", "Patient 406", "Patient 10-T1",  "Patient 10-T2",  "Patient 10-T3", "Patient 15-T1",  "Patient 15-T2",  "Patient 15-T3"]]
    if nn:
        sns.heatmap(data=tFacDF, ax=ax, vmin=0, vmax=1)
    else:
        cmap = sns.color_palette("vlag", as_cmap=True)
        sns.heatmap(data=tFacDF, ax=ax, cmap=cmap, vmin=-1, vmax=1)


def CoH_LogReg_plot(ax, tFac, CoH_Array, numComps):
    """Plot factor weights for donor BC prediction"""
    coord = CoH_Array.dims.index("Patient")
    mode_facs = tFac[1][coord]
    Donor_CoH_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    LR_CoH = LogisticRegression(random_state=0).fit(mode_facs, Donor_CoH_y)
    CoH_comp_weights = pd.DataFrame({"Component": np.arange(1, numComps + 1), "Coefficient": LR_CoH.coef_[0]})
    sns.barplot(data=CoH_comp_weights, x="Component", y="Coefficient", color="k", ax=ax)
    print(LR_CoH.score(mode_facs, Donor_CoH_y))
