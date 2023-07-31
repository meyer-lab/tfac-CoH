import numpy as np
from copy import deepcopy
import seaborn as sns
import pandas as pd
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign
from tensorly.tenalg import khatri_rao
from tensorpack.cmtf import initialize_cp, cp_normalize, calcR2X, mlstsq, sort_factors, tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn import preprocessing
from .flow import get_status_df


def factorTensor(tOrig: np.ndarray, r: int, tol: float=1e-9, maxiter: int=6_000, progress: bool=False, linesearch: bool=True):
    """ Perform CP decomposition. """
    tFac = initialize_cp(tOrig, r)

    acc_pow: float = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
    acc_fail: int = 0  # How many times acceleration have failed
    max_fail: int = 4  # Increase acc_pow with one after max_fail failure

    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]

    R2X_last = -np.inf
    R2X = calcR2X(tFac, tOrig)

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    tq = tqdm(range(maxiter), disable=(not progress))
    for i in tq:
        tFac_old = deepcopy(tFac)

        # Solve on each mode
        for m in range(len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = mlstsq(kr, unfolded[m].T, uniqueInfo[m]).T

        R2X_last = R2X
        R2X = calcR2X(tFac, tOrig)
        assert R2X > 0.0
        tq.set_postfix(R2X=R2X, delta=R2X - R2X_last, refresh=False)

        # Initiate line search
        if linesearch and i % 2 == 0 and i > 5:
            jump = i ** (1.0 / acc_pow)

            # Estimate error with line search
            tFac_ls = deepcopy(tFac)
            tFac_ls.factors = [
                f + (f - tFac_old.factors[ii]) * jump
                for ii, f in enumerate(tFac.factors)
            ]
            R2X_ls = calcR2X(tFac_ls, tOrig)

            if R2X_ls > R2X:
                acc_fail = 0
                tFac = tFac_ls
            else:
                acc_fail += 1

                if acc_fail == max_fail:
                    acc_pow += 1.0
                    acc_fail = 0

        if R2X - R2X_last < tol:
            break

    tFac = cp_normalize(tFac)
    tFac = cp_flip_sign(tFac)

    if r > 1:
        tFac = sort_factors(tFac)
    
    return tFac, R2X


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
lrmodel = LogisticRegressionCV(penalty='elasticnet', solver="saga", max_iter=5000, l1_ratios=[0.9], Cs=[100.0, 1.0, 0.1], cv=cv)


def varyCompPlots(axs: list, compNum: int, data, yDf: pd.DataFrame):
    """Plot 5 fold CV by # components"""
    accDF = pd.DataFrame()
    R2X = np.zeros(compNum)
    comps = np.arange(1, compNum + 1)
    yDf = yDf.set_index("Patient")
    yDf = yDf.loc[data.Patient.values, :]

    y = preprocessing.label_binarize(yDf.Status, classes=['Healthy', 'BC']).flatten()

    for i in comps:
        tFacAllM, R2X[i - 1] = factorTensor(data.values, r=i)
        coord = data.dims.index("Patient")
        mode_facs = tFacAllM[1][coord]

        lrmodel.fit(mode_facs, y)

        scoresTFAC = cross_val_score(lrmodel, mode_facs, y, cv=cv)
        accDF = pd.concat([accDF, pd.DataFrame({"Data Type": "Tensor Factorization", "Components": [i], "Accuracy (10-fold CV)": np.mean(scoresTFAC)})])

    accDF = accDF.reset_index(drop=True)

    sns.lineplot(data=accDF, x="Components", y="Accuracy (10-fold CV)", hue="Data Type", ax=axs[0])
    axs[0].set(xticks=comps, ylim=(0.5, 1))

    axs[1].scatter(comps, R2X, c='k', s=20.)
    axs[1].set(title="R2X", ylabel="Variance Explained", xlabel="Number of Components", ylim=(0, 1), xlim=(0, compNum + 0.5), xticks=np.arange(0, compNum + 1))


def plot_tFac_CoH(ax, tFac, CoH_Array, mode, cbar=False):
    """Plots tensor factorization of cells"""
    mode_facs = tFac.factors[CoH_Array.dims.index(mode)]
    tFacDF = pd.DataFrame(mode_facs, index=CoH_Array.coords[mode], columns=[str(i) for i in range(1, mode_facs.shape[1] + 1)])

    cmap = sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(data=tFacDF.T, ax=ax, cmap=cmap, vmin=-1, vmax=1, cbar=cbar)


def CoH_LogReg_plot(ax, tFac, CoH_Array):
    """Plot factor weights for donor BC prediction"""
    coord = CoH_Array.dims.index("Patient")
    mode_facs = tFac[1][coord]
    status_DF = get_status_df()
    Donor_CoH_y = preprocessing.label_binarize(status_DF.Status, classes=['Healthy', 'BC']).flatten()

    LR_CoH = lrmodel.fit(mode_facs, Donor_CoH_y)
    CoH_comp_weights = pd.DataFrame({"Component": np.arange(1, mode_facs.shape[1] + 1), "Coefficient": LR_CoH.coef_[0]})
    sns.barplot(data=CoH_comp_weights, x="Component", y="Coefficient", color="k", ax=ax)
