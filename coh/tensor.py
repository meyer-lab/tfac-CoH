from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
import tensorly as tl
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorly.cp_tensor import CPTensor, cp_flip_sign
from tensorly.decomposition._cp import initialize_cp
from tensorly.tenalg.einsum_tenalg import khatri_rao
from tensorpack.cmtf import calcR2X, cp_normalize, mlstsq, tqdm


def factorTensor(
    tOrig: np.ndarray,
    r: int,
    tol: float = 1e-9,
    maxiter: int = 6_000,
    progress: bool = False,
    linesearch: bool = True,
):
    """Perform CP decomposition."""
    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]

    tFac = CPTensor(initialize_cp(np.nan_to_num(tOrig), r))

    acc_pow: float = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
    acc_fail: int = 0  # How many times acceleration have failed
    max_fail: int = 4  # Increase acc_pow with one after max_fail failure

    R2X_last = -np.inf
    R2X = calcR2X(tFac, tOrig)

    # Precalculate the missingness patterns
    uniqueInfo = [
        np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded
    ]

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
        gini_idx = giniIndex(tFac.factors[0])
        tFac.factors = [f[:, gini_idx] for f in tFac.factors]
        tFac.weights = tFac.weights[gini_idx]

    tFac.R2X = R2X
    return tFac


def giniIndex(X: np.ndarray) -> np.ndarray:
    """Calculates the Gini Coeff for each component and returns the index rearrangment."""
    X = np.abs(X)
    gini = np.var(X, axis=0) / np.mean(X, axis=0)

    return np.argsort(gini)


def R2Xplot(ax, tensor, compNum: int) -> None:
    """Creates R2X plot for non-neg CP tensor decomposition."""
    varHold = [factorTensor(tensor, i).R2X for i in range(1, compNum + 1)]
    ax.scatter(np.arange(1, compNum + 1), varHold, c="k", s=20.0)
    ax.set(
        title="R2X",
        ylabel="Variance Explained",
        xlabel="Number of Components",
        ylim=(0, 1),
        xlim=(0, compNum + 0.5),
        xticks=np.arange(0, compNum + 1),
    )


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=20)
lrmodel = LogisticRegressionCV(
    penalty="l1", solver="saga", max_iter=5000, tol=1e-6, cv=cv, n_jobs=4,
)


def CoH_LogReg_plot(ax, tFac, CoH_Array, status_DF) -> None:
    """Plot factor weights for donor BC prediction."""
    coord = CoH_Array.dims.index("Patient")
    mode_facs = tFac[1][coord]
    Donor_CoH_y = preprocessing.label_binarize(
        status_DF.Status, classes=["Healthy", "BC"],
    ).flatten()

    LR_CoH = lrmodel.fit(mode_facs, Donor_CoH_y)

    CoH_comp_weights = pd.DataFrame(
        {
            "Component": np.arange(1, mode_facs.shape[1] + 1),
            "Coefficient": LR_CoH.coef_[0],
        },
    )
    sns.barplot(data=CoH_comp_weights, x="Component", y="Coefficient", color="k", ax=ax)


def BC_status_plot(compNum, CoH_Data, ax, status_DF) -> None:
    """Plot 5 fold CV by # components."""
    accDF = pd.DataFrame()
    Donor_CoH_y = preprocessing.label_binarize(
        status_DF.Status, classes=["Healthy", "BC"],
    ).flatten()

    for i in range(1, compNum + 1):
        tFacAllM = factorTensor(CoH_Data.to_numpy(), r=i)
        coord = CoH_Data.dims.index("Patient")
        mode_facs = tFacAllM[1][coord]

        lrmodel.fit(mode_facs, Donor_CoH_y)
        scoresTFAC = np.max(np.mean(lrmodel.scores_[1], axis=0))
        accDF = pd.concat(
            [
                accDF,
                pd.DataFrame(
                    {
                        "Data Type": "Tensor Factorization",
                        "Components": [i],
                        "Accuracy (10-fold CV)": scoresTFAC,
                    },
                ),
            ],
        )

    accDF = accDF.reset_index(drop=True)
    sns.scatterplot(
        data=accDF,
        x="Components",
        y="Accuracy (10-fold CV)",
        hue="Data Type",
        ax=ax,
        color="k",
    )
    ax.set(xticks=np.arange(1, compNum + 1), ylim=(0.5, 1))
            