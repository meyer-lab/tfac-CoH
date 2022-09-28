"""
This creates Figure 1.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from tensorpack.cmtf import cp_normalize
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..tensor import factorTensor,CoH_LogReg_plot, plot_tFac_CoH, make_alldata_DF
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    #make_flow_df()
    #make_CoH_Tensor(just_signal=True)

    num_comps = 12
    
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    cp_normalize(tFacAllM)
    #make_alldata_DF(CoH_Data, PCA=False)
    CoH_LogReg_plot(ax[1], tFacAllM, CoH_Data, num_comps)
    matrix_DF = pd.read_csv(join(path_here, "data/CoH_Matrix.csv"), index_col=0).dropna(axis='columns').set_index("Patient")

    BC_status_plot(15, CoH_Data, matrix_DF, ax[0])

    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Treatment", numComps=num_comps)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Marker", numComps=num_comps)

    CoH_DF = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    BC_scatter(ax[4], CoH_DF, "pSTAT3", "IL10-50ng")
    BC_scatter(ax[5], CoH_DF, "pSTAT5", "IL2-50ng")

    return f


def BC_status_plot(compNum, CoH_Data, matrixDF, ax):
    """Plot 5 fold CV by # components"""
    accDF = pd.DataFrame()
    status_DF = pd.read_csv(join(path_here, "data/Patient_Status.csv"), index_col=0)
    matrixDF = matrixDF.values
    lb = preprocessing.LabelBinarizer()
    Donor_CoH_y = lb.fit_transform(status_DF.Status).ravel()
    cv = StratifiedKFold(n_splits=5)
    model = LogisticRegression()
    scoresPCA = cross_val_score(model, matrixDF, Donor_CoH_y, cv=cv)
    for i in range(5, compNum + 1):
        if i != 14:
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
            accDF = pd.concat([accDF, pd.DataFrame({"Data Type": "Tensor Factorization", "Components": [i], "Accuracy (5-fold CV)": np.mean(scoresTFAC)})])
            accDF = pd.concat([accDF, pd.DataFrame({"Data Type": "All Data", "Components": [i], "Accuracy (5-fold CV)": np.mean(scoresPCA)})])
    accDF = accDF.reset_index(drop=True)
    sns.lineplot(data=accDF, x="Components", y="Accuracy (5-fold CV)", hue="Data Type", ax=ax)
    ax.set(xticks = np.arange(5, compNum + 1))


def BC_scatter(ax, CoH_DF, marker, cytokine):
    """Scatters specific responses"""
    hist_DF = CoH_DF.loc[(CoH_DF.Treatment == cytokine) & (CoH_DF.Marker == marker)]
    hist_DF["Status"] = hist_DF.replace({"Patient": status_dict}).Patient.values
    sns.histplot(data=hist_DF, x="Mean", hue="Status", ax=ax)
    ax.set(title=marker + " in response to " + cytokine, xlabel=marker, ylabel="Count")


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
                "Patient 4": "BC", 
                "Patient 8": "BC", 
                "Patient 406": "BC", 
                "Patient 10-T1": "BC",
                "Patient 10-T2": "BC",  
                "Patient 10-T3": "BC", 
                "Patient 15-T1": "BC",  
                "Patient 15-T2": "BC",  
                "Patient 15-T3": "BC",
                "Patient 19186-2": "BC",
                "Patient 19186-3": "BC",
                "Patient 19186-14": "BC",
                "Patient 21368-3": "BC",
                "Patient 21368-4": "BC"}
