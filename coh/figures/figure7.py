"""
This creates Figure 1.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import cp_normalize, perform_CP
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..tensor import CoH_LogReg_plot
from ..flow import make_flow_df, make_CoH_Tensor

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    #make_flow_df(foldChange=True)
    #make_CoH_Tensor(just_signal=True, foldChange=True)
    num_comps = 15

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Tensor_DataSet_FC.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    cp_normalize(tFacAllM)

    #makePCA_df(CoH_Data)
    CoH_LogReg_plot(ax[0], tFacAllM, CoH_Data, num_comps)
    PCAdf = pd.read_csv(join(path_here, "data/CoH_PCA.csv")).dropna(axis='columns').drop("Unnamed: 0", axis=1).set_index("Patient")

    tfacDF = plot_tFac_CoH(ax[1], tFacAllM, CoH_Data, "Patient", numComps=num_comps)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Time", numComps=num_comps)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Treatment", numComps=num_comps)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Cell", numComps=num_comps)
    plot_tFac_CoH(ax[5], tFacAllM, CoH_Data, "Marker", numComps=num_comps)

    PCA_X = PCAdf.values
    TFAC_X = tfacDF.transpose().values
    Donor_CoH_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cv = StratifiedKFold(n_splits=5)
    model = LogisticRegression()
    #print(PCA_X)
    scoresPCA = cross_val_score(model, PCA_X, Donor_CoH_y, cv=cv)
    scoresTFAC = cross_val_score(model, TFAC_X, Donor_CoH_y, cv=cv)
    print(scoresPCA)
    print(scoresTFAC)

    return f


def factorTensor(tensor, numComps):
    """ Takes Tensor, and mask and returns tensor factorized form. """
    tfac = perform_CP(tensor, numComps, tol=1e-7, maxiter=1000)
    R2X = tfac.R2X
    tfac = cp_flip_sign(tfac)
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
        tFacDF = tFacDF[["Patient 35", "Patient 43", "Patient 44", "Patient 45", "Patient 52", "Patient 54", "Patient 56", "Patient 58", "Patient 63", "Patient 66", "Patient 70", "Patient 79", "Patient 4", "Patient 8", "Patient 406", "Patient 10-T1",  "Patient 10-T2",  "Patient 10-T3", "Patient 15-T1",  "Patient 15-T2",  "Patient 15-T3"]]
    sns.heatmap(data=tFacDF, ax=ax, cmap=cmap, vmin=-1, vmax=1)
    return tFacDF


def makePCA_df(TensorArray):
    """Returns PCA with score and loadings of COH DataSet"""
    DF = TensorArray.to_dataframe(name="value").reset_index()
    #DF = DF.loc[(DF.Patient != "Patient 4") & (DF.Patient != "Patient 8") & (DF.Patient != "Patient 406")]
    PCAdf = pd.DataFrame()
    for patient in DF.Patient.unique():
        patientDF = DF.loc[DF.Patient == patient]
        patientRow = pd.DataFrame({"Patient": [patient]})
        for time in DF.Time.unique():
            for treatment in DF.Treatment.unique():
                for marker in DF.Marker.unique():
                    for cell in DF.Cell.unique():
                        uniqueDF = patientDF.loc[(patientDF.Time == time) & (patientDF.Marker == marker) & (patientDF.Treatment == treatment) & (patientDF.Cell == cell)]
                        patientRow[time + "_" + treatment + "_"+ marker + "_"+ cell] = uniqueDF.value.values
        PCAdf = pd.concat([PCAdf, patientRow])
    PCAdf.to_csv(join(path_here, "data/CoH_PCA.csv"))
