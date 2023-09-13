"""
This creates Figure 2I, Characterization of component dependency when classifying BC patients
"""
import pickle
import seaborn as sns
import pandas as pd
import itertools
import numpy as np
from .common import subplotLabel, getSetup
from ..flow import make_CoH_Tensor, get_status_df
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))

    # Add subplot labels
    subplotLabel(ax)
    CoH_Data = make_CoH_Tensor(just_signal=True)

    with open("./coh/data/signaling.pkl", "rb") as ff:
        tFacAllM = pickle.load(ff)  # 12 component

    CoH_comp_scan_plot(ax[0], tFacAllM, CoH_Data, get_status_df())

    return f


def CoH_comp_scan_plot(ax, tFac, CoH_Array, status_DF):
    """Plot factor weights for donor BC prediction"""
    lrmodel = LogisticRegression(penalty="l2", C=1000.0)
    coord = CoH_Array.dims.index("Patient")
    mode_facs = tFac[1][coord]

    Donor_CoH_y = preprocessing.label_binarize(
        status_DF.Status, classes=["Healthy", "BC"]
    ).flatten()
    all_comps = np.arange(0, mode_facs.shape[1])
    Acc_DF = pd.DataFrame()

    for comps in itertools.product(all_comps, all_comps):
        if comps[0] == comps[1]:
            compFacs = mode_facs[:, comps[0]][:, np.newaxis]
        else:
            compFacs = mode_facs[:, [comps[0], comps[1]]]

        LR_CoH = lrmodel.fit(compFacs, Donor_CoH_y)
        acc = LR_CoH.score(compFacs, Donor_CoH_y)
        Acc_DF = pd.concat(
            [
                Acc_DF,
                pd.DataFrame(
                    {
                        "Component 1": "Comp. " + str(comps[0] + 1),
                        "Component 2": "Comp. " + str(comps[1] + 1),
                        "Accuracy": [acc],
                    }
                ),
            ]
        )

    Acc_DF = Acc_DF.pivot_table(
        index="Component 1", columns="Component 2", values="Accuracy", sort=False
    )
    sns.heatmap(
        data=Acc_DF,
        vmin=0.5,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Accuracy 10-fold CV"},
        ax=ax,
    )
    ax.set(xlabel="First Component", ylabel="Second Component")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
