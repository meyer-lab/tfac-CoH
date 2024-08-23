"""This creates Figure 8, an examination of tucker decomposition for response data."""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from tensorpack.decomposition import Decomposition
from tensorpack.plot import tucker_reduction
from tensorpack.tucker import tucker_decomp

from ..flow import get_status_df, make_CoH_Tensor
from ..flow_rec import get_status_rec_df, make_CoH_Tensor_rec
from ..tensor import lrmodel
from .common import getSetup, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Response Data
    CoH_Data = make_CoH_Tensor(just_signal=True)

    tuck_decomp = Decomposition(CoH_Data, method=tucker_decomp)
    tuck_decomp.perform_tucker()

    cp_decomp = Decomposition(CoH_Data.to_numpy(), max_rr=12)
    
    tucker_reduction(ax[0], tuck_decomp, cp_decomp)
    BC_status_plot_tuck(tuck_decomp, ax[1], get_status_df())

    # Receptor data
    CoH_Data_R = make_CoH_Tensor_rec()
    tuck_decomp_R = Decomposition(CoH_Data_R, method=tucker_decomp)
    tuck_decomp.perform_tucker()

    cp_decomp_R = Decomposition(CoH_Data_R.to_numpy(), max_rr=8)
    
    tucker_reduction(ax[2], tuck_decomp_R, cp_decomp_R)
    BC_status_plot_tuck(tuck_decomp_R, ax[3], get_status_rec_df())

    return f


def BC_status_plot_tuck(tuck_decomp, ax, status_DF) -> None:
    """Plot 5 fold CV by # components."""
    accDF = pd.DataFrame()
    Donor_CoH_y = preprocessing.label_binarize(
        status_DF.Status, classes=["Healthy", "BC"],
    ).flatten()
    xticks = []

    for i in range(1, len(tuck_decomp.TuckRank)):
        mode_facs = tuck_decomp.Tucker[i][1][0]

        lrmodel.fit(mode_facs, Donor_CoH_y)
        scoresTFAC = np.max(np.mean(lrmodel.scores_[1], axis=0))
        accDF = pd.concat(
            [
                accDF,
                pd.DataFrame(
                    {
                        "Components": [i],
                        "Accuracy (10-fold CV)": scoresTFAC,
                    },
                ),
            ],
        )
        xticks.append(" ".join(str(x) for x in tuck_decomp.TuckRank[i]))

    accDF = accDF.reset_index(drop=True)
    sns.scatterplot(
        data=accDF,
        x="Components",
        y="Accuracy (10-fold CV)",
        ax=ax,
        color="k",
    )
    ax.set(xticks=np.arange(1, len(xticks) + 1), xticklabels=xticks, ylim=(0.5, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
