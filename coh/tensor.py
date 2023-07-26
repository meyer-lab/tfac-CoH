import numpy as np
import seaborn as sns
import pandas as pd
from collections import OrderedDict
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import perform_CP, cp_normalize
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn import preprocessing
from os.path import join, dirname

path_here = dirname(dirname(__file__))


def factorTensor(tensor, numComps):
    """ Takes Tensor, and mask and returns tensor factorized form. """
    tfac = perform_CP(tensor, numComps, tol=1e-9, maxiter=1000)
    R2X = tfac.R2X
    tfac = cp_normalize(tfac)
    tfac = cp_flip_sign(tfac)
    return tfac, R2X


def R2Xplot(ax, tensor, compNum: int):
    """Creates R2X plot for non-neg CP tensor decomposition"""
    varHold = [factorTensor(tensor, i)[1] for i in range(1, compNum + 1)]
    ax.scatter(np.arange(1, compNum + 1), varHold, c='k', s=20.)
    ax.set(title="R2X", ylabel="Variance Explained", xlabel="Number of Components", ylim=(0, 1), xlim=(0, compNum + 0.5), xticks=np.arange(0, compNum + 1))


def plot_tFac_CoH(ax, tFac, CoH_Array, mode, numComps=3, nn=False, rec=False, cbar=True):
    """Plots tensor factorization of cells"""
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame()

    for i in range(0, numComps):
        tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, i], "Component": (i + 1), mode: mode_labels})])

    tFacDF = pd.pivot(tFacDF, index="Component", columns=mode, values="Component_Val")
    if mode == "Patient":
        if rec:
            tFacDF = tFacDF[get_status_dict_rec().keys()]
        else:
            tFacDF = tFacDF[get_status_dict().keys()]
    if nn:
        sns.heatmap(data=tFacDF, ax=ax, vmin=0, vmax=1)
    else:
        cmap = sns.color_palette("vlag", as_cmap=True)
        sns.heatmap(data=tFacDF, ax=ax, cmap=cmap, vmin=-1, vmax=1, cbar=cbar)


def CoH_LogReg_plot(ax, tFac, CoH_Array, numComps):
    """Plot factor weights for donor BC prediction"""
    coord = CoH_Array.dims.index("Patient")
    mode_facs = tFac[1][coord]
    status_DF = pd.read_csv(join(path_here, "coh/data/Patient_Status.csv"), index_col=0)
    Donor_CoH_y = preprocessing.label_binarize(status_DF.Status, classes=['Healthy', 'BC']).flatten()

    LR_CoH = LogisticRegressionCV(random_state=0, penalty='l2', max_iter=5000).fit(mode_facs, Donor_CoH_y)
    CoH_comp_weights = pd.DataFrame({"Component": np.arange(1, numComps + 1), "Coefficient": LR_CoH.coef_[0]})
    sns.barplot(data=CoH_comp_weights, x="Component", y="Coefficient", color="k", ax=ax)


def plot_PCA(ax):
    """Plots CoH PCA"""
    DF = pd.read_csv(join(path_here, "data/CoH_PCA.csv")).set_index("Patient").drop("Unnamed: 0", axis=1)
    pcaMat = DF.to_numpy()
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    pcaMat = scaler.fit_transform(np.nan_2_num(pcaMat))
    scores = pca.fit_transform(pcaMat)
    loadings = pca.components_

    scoresDF = pd.DataFrame({"Patient": DF.index.values, "Component 1": scores[:, 0], "Component 2": scores[:, 1]})
    loadingsDF = pd.DataFrame()
    for i, col in enumerate(DF.columns):
        vars = col.split("_")
        loadingsDF = pd.concat([loadingsDF, pd.DataFrame({"Time": [vars[0]], "Treatment": vars[1], "Marker": vars[2], "Cell": vars[3], "Component 1": loadings[0, i], "Component 2": loadings[1, i]})])

    sns.scatterplot(data=scoresDF, hue="Patient", x="Component 1", y="Component 2", ax=ax[0])
    sns.scatterplot(data=loadingsDF, x="Component 1", y="Component 2", hue="Treatment", style="Cell", size="Marker", ax=ax[1])


def BC_status_plot(compNum, CoH_Data, ax, rec=False):
    """Plot 5 fold CV by # components"""
    accDF = pd.DataFrame()
    if rec:
        status_DF = pd.read_csv(join(path_here, "coh/data/Patient_Status_Rec.csv"), index_col=0)
    else:
        status_DF = pd.read_csv(join(path_here, "coh/data/Patient_Status.csv"), index_col=0)
    Donor_CoH_y = preprocessing.label_binarize(status_DF.Status, classes=['Healthy', 'BC']).flatten()
    cv = RepeatedStratifiedKFold(n_splits=10, random_state=42)
    model = LogisticRegressionCV(penalty='l2', max_iter=5000)
    for i in range(1, compNum + 1):
        print(i)
        tFacAllM, _ = factorTensor(CoH_Data.values, numComps=i)
        mode_labels = CoH_Data["Patient"]
        coord = CoH_Data.dims.index("Patient")
        mode_facs = tFacAllM[1][coord]
        tFacDF = pd.DataFrame()
        for j in range(0, i):
            tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, j], "Component": (j + 1), "Patient": mode_labels})])

        tFacDF = pd.pivot(tFacDF, index="Component", columns="Patient", values="Component_Val")
        tFacDF = tFacDF[status_DF.Patient]
        TFAC_X = tFacDF.transpose().values
        model = LogisticRegressionCV(penalty='l2', max_iter=5000)
        scoresTFAC = cross_val_score(model, TFAC_X, Donor_CoH_y, cv=cv)
        accDF = pd.concat([accDF, pd.DataFrame({"Data Type": "Tensor Factorization", "Components": [i], "Accuracy (10-fold CV)": np.mean(scoresTFAC)})])
    accDF = accDF.reset_index(drop=True)
    sns.lineplot(data=accDF, x="Components", y="Accuracy (10-fold CV)", hue="Data Type", ax=ax)
    ax.set(xticks=np.arange(1, compNum + 1), ylim=(0.5, 1))


def get_status_dict():
    """Returns status dictionary"""
    return OrderedDict([("Patient 26", "Healthy"),
                        ("Patient 28", "Healthy"),
                        ("Patient 30", "Healthy"),
                        ("Patient 34", "Healthy"),
                        ("Patient 35", "Healthy"),
                        ("Patient 43", "Healthy"),
                        ("Patient 44", "Healthy"),
                        ("Patient 45", "Healthy"),
                        ("Patient 52", "Healthy"),
                        ("Patient 52A", "Healthy"),
                        ("Patient 54", "Healthy"),
                        ("Patient 56", "Healthy"),
                        ("Patient 58", "Healthy"),
                        ("Patient 60", "Healthy"),
                        ("Patient 61", "Healthy"),
                        ("Patient 62", "Healthy"),
                        ("Patient 63", "Healthy"),
                        ("Patient 66", "Healthy"),
                        ("Patient 68", "Healthy"),
                        ("Patient 69", "Healthy"),
                        ("Patient 70", "Healthy"),
                        ("Patient 79", "Healthy"),
                        ("Patient 19186-4", "BC"),
                        ("Patient 19186-8", "BC"),
                        ("Patient 406", "BC"),
                        ("Patient 19186-10-T1", "BC"),
                        ("Patient 19186-10-T2", "BC"),
                        ("Patient 19186-10-T3", "BC"),
                        ("Patient 19186-15-T1", "BC"),
                        ("Patient 19186-15-T2", "BC"),
                        ("Patient 19186-15-T3", "BC"),
                        ("Patient 19186-2", "BC"),
                        ("Patient 19186-3", "BC"),
                        ("Patient 19186-14", "BC"),
                        ("Patient 21368-3", "BC"),
                        ("Patient 21368-4", "BC")])


def get_status_dict_rec():
    """Returns status dictionary"""
    return OrderedDict([("Patient 26", "Healthy"),
                        ("Patient 28", "Healthy"),
                        ("Patient 30", "Healthy"),
                        ("Patient 34", "Healthy"),
                        ("Patient 35", "Healthy"),
                        ("Patient 43", "Healthy"),
                        ("Patient 44", "Healthy"),
                        ("Patient 45", "Healthy"),
                        ("Patient 52", "Healthy"),
                        ("Patient 52A", "Healthy"),
                        ("Patient 54", "Healthy"),
                        ("Patient 56", "Healthy"),
                        ("Patient 58", "Healthy"),
                        ("Patient 60", "Healthy"),
                        ("Patient 61", "Healthy"),
                        ("Patient 62", "Healthy"),
                        ("Patient 63", "Healthy"),
                        ("Patient 66", "Healthy"),
                        ("Patient 68", "Healthy"),
                        ("Patient 69", "Healthy"),
                        ("Patient 70", "Healthy"),
                        ("Patient 79", "Healthy"),
                        ("Patient 19186-4", "BC"),
                        ("Patient 19186-8", "BC"),
                        ("Patient 19186-10-T1", "BC"),
                        ("Patient 19186-10-T2", "BC"),
                        ("Patient 19186-10-T3", "BC"),
                        ("Patient 19186-15-T1", "BC"),
                        ("Patient 19186-15-T2", "BC"),
                        ("Patient 19186-15-T3", "BC"),
                        ("Patient 19186-2", "BC"),
                        ("Patient 19186-3", "BC"),
                        ("Patient 19186-12", "BC"),
                        ("Patient 19186-14", "BC"),
                        ("Patient 21368-3", "BC"),
                        ("Patient 21368-4", "BC")])
