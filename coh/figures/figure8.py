"""
This creates Figure 8, an examination of tucker decomposition for response data.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from ..flow import make_CoH_Tensor, get_status_df
from ..flow_rec import make_CoH_Tensor_rec, get_status_rec_df
from tensorpack.tucker import tucker_decomp
from tensorpack.plot import tucker_reduction
from tensorpack.decomposition import Decomposition
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import preprocessing
    
def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    #Response Data
    CoH_Data = make_CoH_Tensor(just_signal=True)
    CoH_Data = np.nan_to_num(CoH_Data)
    
    tuck_decomp = Decomposition(CoH_Data, method=tucker_decomp)
    cp_decomp = Decomposition(CoH_Data, max_rr=15)

    tuck_decomp.perform_tucker()
    tucker_reduction(ax[0], tuck_decomp, cp_decomp)
    BC_status_plot_tuck(tuck_decomp, ax[1], get_status_df())

    # Receptor data
    CoH_Data_R = make_CoH_Tensor_rec()
    CoH_Data_R = np.nan_to_num(CoH_Data_R)
    
    tuck_decomp_R = Decomposition(CoH_Data_R, method=tucker_decomp)
    cp_decomp_R = Decomposition(CoH_Data_R, max_rr=8)

    tuck_decomp.perform_tucker()
    tucker_reduction(ax[2], tuck_decomp_R, cp_decomp_R)
    BC_status_plot_tuck(tuck_decomp_R, ax[3], get_status_rec_df())

    return f


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=20)
lrmodel = LogisticRegressionCV(penalty="l1", solver="saga", max_iter=5000, tol=1e-6, cv=cv)

def CoH_tuck_LogReg_plot(ax, tFac, CoH_Array, status_DF):
    """Plot factor weights for donor BC prediction"""
    mode_facs = tFac[0]
    Donor_CoH_y = preprocessing.label_binarize(status_DF.Status, classes=['Healthy', 'BC']).flatten()

    LR_CoH = lrmodel.fit(mode_facs, Donor_CoH_y)
    print(np.max(np.mean(lrmodel.scores_[1], axis=0)))
    CoH_comp_weights = pd.DataFrame({"Component": np.arange(1, mode_facs.shape[1] + 1), "Coefficient": LR_CoH.coef_[0]})
    sns.barplot(data=CoH_comp_weights, x="Component", y="Coefficient", color="k", ax=ax)


def BC_status_plot_tuck(tuck_decomp, ax, status_DF):
    """Plot 5 fold CV by # components"""
    accDF = pd.DataFrame()
    Donor_CoH_y = preprocessing.label_binarize(status_DF.Status, classes=['Healthy', 'BC']).flatten()
    xticks = []

    for i in range(1, len(tuck_decomp.TuckRank)):
        mode_facs = tuck_decomp.Tucker[i][1][0]

        lrmodel.fit(mode_facs, Donor_CoH_y)
        scoresTFAC = np.max(np.mean(lrmodel.scores_[1], axis=0))
        accDF = pd.concat([accDF, pd.DataFrame({"Data Type": "Tensor Factorization", "Components": [i], "Accuracy (10-fold CV)": scoresTFAC})])
        xticks.append(" ".join(str(x) for x in tuck_decomp.TuckRank[i]))

    accDF = accDF.reset_index(drop=True)
    sns.scatterplot(data=accDF, x="Components", y="Accuracy (10-fold CV)", hue="Data Type", ax=ax, color='k')
    ax.set(xticks=np.arange(1, len(xticks) + 1), xticklabels=xticks, ylim=(0.5, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
