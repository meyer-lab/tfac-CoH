from os.path import join, dirname
import numpy as np
import pandas as pd
import tensorly as tl
import seaborn as sns
from tensorly.cp_tensor import cp_flip_sign, cp_to_tensor
from tensorly.decomposition import parafac
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tensorpack.cmtf import cp_normalize, calcR2X
from sklearn import preprocessing

from tlviz.model_evaluation import core_consistency

path_here = dirname(dirname(__file__))


def factorTensor(tensor, numComps):
    """ Takes Tensor, and mask and returns tensor factorized form. """
    tfac = parafac(np.nan_to_num(tensor), numComps, mask=np.isnan(tensor), verbose=True, svd_mask_repeats=0, linesearch=True)
    R2X = calcR2X(tensor, tfac)
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


def core_cons_plot(ax, tensor, compNum):
    """Creates R2X plot for non-neg CP tensor decomposition"""
    ccHold = np.zeros(compNum)
    for i in range(1, compNum + 1):
        print(i)
        tfac, _ = factorTensor(tensor, i)
        X = cp_to_tensor(tfac)
        CC = core_consistency(tfac, X, True)
        ccHold[i - 1] = CC / 100

    ax.scatter(np.arange(1, compNum + 1), ccHold, c='k', s=20.)
    ax.set(title="Core Consistency", ylabel="Core Consistency", xlabel="Number of Components", ylim=(0, 1), xlim=(0, compNum + 0.5), xticks=np.arange(0, compNum + 1))


def calcR2X(tensorIn, tensorFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - tensorIn)
    return 1.0 - tErr / np.nanvar(tensorIn)


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
            tFacDF = tFacDF[["Patient 26",
                             "Patient 28",
                             "Patient 30",
                             "Patient 34",
                             "Patient 35",
                             "Patient 43",
                             "Patient 44",
                             "Patient 45",
                             "Patient 52",
                             "Patient 52A",
                             "Patient 54",
                             "Patient 56",
                             "Patient 58",
                             "Patient 60",
                             "Patient 61",
                             "Patient 62",
                             "Patient 63",
                             "Patient 66",
                             "Patient 68",
                             "Patient 69",
                             "Patient 70",
                             "Patient 79",
                             "Patient 19186-2",
                             "Patient 19186-3",
                             "Patient 19186-4",
                             "Patient 19186-8",
                             "Patient 19186-10-T1",
                             "Patient 19186-10-T2",
                             "Patient 19186-10-T3",
                             "Patient 19186-15-T1",
                             "Patient 19186-15-T2",
                             "Patient 19186-15-T3",
                             "Patient 19186-12",
                             "Patient 19186-14",
                             "Patient 21368-3",
                             "Patient 21368-4"]]
        else:
            tFacDF = tFacDF[["Patient 26",
                             "Patient 28",
                             "Patient 30",
                             "Patient 34",
                             "Patient 35",
                             "Patient 43",
                             "Patient 44",
                             "Patient 45",
                             "Patient 52",
                             "Patient 52A",
                             "Patient 54",
                             "Patient 56",
                             "Patient 58",
                             "Patient 60",
                             "Patient 61",
                             "Patient 62",
                             "Patient 63",
                             "Patient 66",
                             "Patient 68",
                             "Patient 69",
                             "Patient 70",
                             "Patient 79",
                             "Patient 19186-2",
                             "Patient 19186-3",
                             "Patient 19186-4",
                             "Patient 19186-8",
                             "Patient 19186-10-T1",
                             "Patient 19186-10-T2",
                             "Patient 19186-10-T3",
                             "Patient 19186-15-T1",
                             "Patient 19186-15-T2",
                             "Patient 19186-15-T3",
                             "Patient 19186-14",
                             "Patient 21368-3",
                             "Patient 21368-4",
                             "Patient 406"]]
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

    LR_CoH = LogisticRegression(random_state=0).fit(mode_facs, Donor_CoH_y)
    CoH_comp_weights = pd.DataFrame({"Component": np.arange(1, numComps + 1), "Coefficient": LR_CoH.coef_[0]})
    sns.barplot(data=CoH_comp_weights, x="Component", y="Coefficient", color="k", ax=ax)


def make_alldata_DF(TensorArray, PCA=True, foldChange=False, basal=False):
    """Returns PCA with score and loadings of COH DataSet"""
    DF = TensorArray.to_dataframe(name="value").reset_index()
    if PCA:
        status_DF = pd.read_csv(join(path_here, "coh/data/Patient_Status.csv"), index_col=0)
        healthy_patients = status_DF.loc[status_DF.Status == "Healthy"]
        DF = DF.loc[DF.Patient.isin(healthy_patients)]
    PCAdf = pd.DataFrame()
    for patient in DF.Patient.unique():
        patientDF = DF.loc[DF.Patient == patient]
        patientRow = pd.DataFrame({"Patient": [patient]})
        for time in DF.Time.unique():
            for treatment in DF.Treatment.unique():
                for marker in DF.Marker.unique():
                    for cell in DF.Cell.unique():
                        uniqueDF = patientDF.loc[(patientDF.Time == time) & (patientDF.Marker == marker) & (patientDF.Treatment == treatment) & (patientDF.Cell == cell)]
                        patientRow[time + "_" + treatment + "_" + marker + "_" + cell] = uniqueDF.value.values
        PCAdf = pd.concat([PCAdf, patientRow])
    if PCA:
        PCAdf.to_csv(join(path_here, "coh/data/CoH_PCA.csv"))
    else:
        if foldChange:
            PCAdf.to_csv(join(path_here, "coh/data/CoH_Matrix_FC.csv"))
        elif basal:
            PCAdf.to_csv(join(path_here, "coh/data/CoH_Matrix_Basal.csv"))
        else:
            PCAdf.to_csv(join(path_here, "coh/data/CoH_Matrix.csv"))


def make_allrec_DF(RecArray):
    """Makes all data DF for rec data"""
    DF = RecArray.to_dataframe(name="value").reset_index()
    PCAdf = pd.DataFrame()
    for patient in DF.Patient.unique():
        patientDF = DF.loc[DF.Patient == patient]
        patientRow = pd.DataFrame({"Patient": [patient]})
        for marker in DF.Marker.unique():
            for cell in DF.Cell.unique():
                uniqueDF = patientDF.loc[(patientDF.Marker == marker) & (patientDF.Cell == cell)]
                patientRow[marker + "_" + cell] = uniqueDF.value.values
        PCAdf = pd.concat([PCAdf, patientRow])
    PCAdf.to_csv(path_here + "/coh/data/CoH_Matrix_Rec.csv")


def plot_PCA(ax):
    """Plots CoH PCA"""
    DF = pd.read_csv(join(path_here, "data/CoH_PCA.csv")).set_index("Patient").drop("Unnamed: 0", axis=1)
    pcaMat = DF.to_numpy()
    pca = PCA(n_components=2)
    scaler = preprocessing.StandardScaler()
    pcaMat = scaler.fit_transform(np.nan_to_num(pcaMat))
    scores = pca.fit_transform(pcaMat)
    loadings = pca.components_

    scoresDF = pd.DataFrame({"Patient": DF.index.values, "Component 1": scores[:, 0], "Component 2": scores[:, 1]})
    loadingsDF = pd.DataFrame()
    for i, col in enumerate(DF.columns):
        vars = col.split("_")
        loadingsDF = pd.concat([loadingsDF, pd.DataFrame({"Time": [vars[0]], "Treatment": vars[1], "Marker": vars[2], "Cell": vars[3], "Component 1": loadings[0, i], "Component 2": loadings[1, i]})])

    sns.scatterplot(data=scoresDF, hue="Patient", x="Component 1", y="Component 2", ax=ax[0])
    sns.scatterplot(data=loadingsDF, x="Component 1", y="Component 2", hue="Treatment", style="Cell", size="Marker", ax=ax[1])


def prediction_model(X, y):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model = LogisticRegression()
    scores = cross_val_score(model, X, y, cv=cv)
    return scores


def BC_status_plot(compNum, CoH_Data, ax, basal=False):
    """Plot 5 fold CV by # components"""
    accDF = pd.DataFrame()
    status_DF = pd.read_csv(join(path_here, "coh/data/Patient_Status.csv"), index_col=0)
    Donor_CoH_y = preprocessing.label_binarize(status_DF.Status, classes=['Healthy', 'BC']).flatten()
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model = LogisticRegression()
    if basal:
        start_val = 1
    else:
        start_val = 1
    for i in range(start_val, compNum + 1):
        tFacAllM, _ = factorTensor(CoH_Data.values, numComps=i)
        cp_normalize(tFacAllM)
        mode_labels = CoH_Data["Patient"]
        coord = CoH_Data.dims.index("Patient")
        mode_facs = tFacAllM[1][coord]
        tFacDF = pd.DataFrame()
        for j in range(0, i):
            tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, j], "Component": (j + 1), "Patient": mode_labels})])

        tFacDF = pd.pivot(tFacDF, index="Component", columns="Patient", values="Component_Val")
        tFacDF = tFacDF[status_DF.Patient]
        TFAC_X = tFacDF.transpose().values
        model = LogisticRegression()
        scoresTFAC = cross_val_score(model, TFAC_X, Donor_CoH_y, cv=cv)
        accDF = pd.concat([accDF, pd.DataFrame({"Data Type": "Tensor Factorization", "Components": [i], "Accuracy (10-fold CV)": np.mean(scoresTFAC)})])
    accDF = accDF.reset_index(drop=True)
    sns.lineplot(data=accDF, x="Components", y="Accuracy (10-fold CV)", hue="Data Type", ax=ax)
    ax.set(xticks=np.arange(start_val, compNum + 1), ylim=(0.5, 1))


def BC_status_plot_rec(compNum, CoH_Data, matrixDF, ax):
    """Plot 5 fold CV by # components"""
    accDF = pd.DataFrame()
    status_DF = pd.read_csv(join(path_here, "coh/data/Patient_Status_Rec.csv"), index_col=0)
    Donor_CoH_y = preprocessing.label_binarize(status_DF.Status, classes=['Healthy', 'BC']).flatten()
    matrixDF = matrixDF.values
    scoresPCA = prediction_model(matrixDF, Donor_CoH_y)
    start_val = 1
    for i in range(start_val, compNum + 1):
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
        scoresTFAC = prediction_model(TFAC_X, Donor_CoH_y)
        accDF = pd.concat([accDF, pd.DataFrame({"Data Type": "Tensor Factorization", "Components": [i], "Accuracy (10-fold CV)": np.mean(scoresTFAC)})])
        accDF = pd.concat([accDF, pd.DataFrame({"Data Type": "All Data", "Components": [i], "Accuracy (10-fold CV)": np.mean(scoresPCA)})])
    accDF = accDF.reset_index(drop=True)
    sns.lineplot(data=accDF, x="Components", y="Accuracy (10-fold CV)", hue="Data Type", ax=ax)
    ax.set(xticks=np.arange(start_val, compNum + 1))


status_dict = {"Patient 26": "Healthy",
               "Patient 28": "Healthy",
               "Patient 30": "Healthy",
               "Patient 34": "Healthy",
               "Patient 35": "Healthy",
               "Patient 43": "Healthy",
               "Patient 44": "Healthy",
               "Patient 45": "Healthy",
               "Patient 52": "Healthy",
               "Patient 52A": "Healthy",
               "Patient 54": "Healthy",
               "Patient 56": "Healthy",
               "Patient 58": "Healthy",
               "Patient 60": "Healthy",
               "Patient 61": "Healthy",
               "Patient 62": "Healthy",
               "Patient 63": "Healthy",
               "Patient 66": "Healthy",
               "Patient 68": "Healthy",
               "Patient 69": "Healthy",
               "Patient 70": "Healthy",
               "Patient 79": "Healthy",
               "Patient 19186-4": "BC",
               "Patient 19186-8": "BC",
               "Patient 406": "BC",
               "Patient 19186-10-T1": "BC",
               "Patient 19186-10-T2": "BC",
               "Patient 19186-10-T3": "BC",
               "Patient 19186-15-T1": "BC",
               "Patient 19186-15-T2": "BC",
               "Patient 19186-15-T3": "BC",
               "Patient 19186-2": "BC",
               "Patient 19186-3": "BC",
               "Patient 19186-14": "BC",
               "Patient 21368-3": "BC",
               "Patient 21368-4": "BC"}


status_dict_rec = {"Patient 26": "Healthy",
               "Patient 28": "Healthy",
               "Patient 30": "Healthy",
               "Patient 34": "Healthy",
               "Patient 35": "Healthy",
               "Patient 43": "Healthy",
               "Patient 44": "Healthy",
               "Patient 45": "Healthy",
               "Patient 52": "Healthy",
               "Patient 52A": "Healthy",
               "Patient 54": "Healthy",
               "Patient 56": "Healthy",
               "Patient 58": "Healthy",
               "Patient 60": "Healthy",
               "Patient 61": "Healthy",
               "Patient 62": "Healthy",
               "Patient 63": "Healthy",
               "Patient 66": "Healthy",
               "Patient 68": "Healthy",
               "Patient 69": "Healthy",
               "Patient 70": "Healthy",
               "Patient 79": "Healthy",
               "Patient 19186-4": "BC",
               "Patient 19186-8": "BC",
               "Patient 19186-10-T1": "BC",
               "Patient 19186-10-T2": "BC",
               "Patient 19186-10-T3": "BC",
               "Patient 19186-15-T1": "BC",
               "Patient 19186-15-T2": "BC",
               "Patient 19186-15-T3": "BC",
               "Patient 19186-2": "BC",
               "Patient 19186-3": "BC",
               "Patient 19186-12": "BC",
               "Patient 19186-14": "BC",
               "Patient 21368-3": "BC",
               "Patient 21368-4": "BC"}


def get_status_dict():
    """Returns status dictionary"""
    return status_dict


def get_status_dict_rec():
    """Returns status dictionary"""
    return status_dict_rec