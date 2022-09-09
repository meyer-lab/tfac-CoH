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
from ..tensor import factorTensor,CoH_LogReg_plot, plot_tFac_CoH
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    #make_flow_df()
    #make_CoH_Tensor(just_signal=True)

    num_comps = 10
    
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    cp_normalize(tFacAllM)
    #makePCA_df(CoH_Data)
    #CoH_LogReg_plot(ax[1], tFacAllM, CoH_Data, num_comps)
    PCAdf = pd.read_csv(join(path_here, "data/CoH_PCA.csv")).dropna(axis='columns').drop("Unnamed: 0", axis=1).set_index("Patient")

    #BC_status_plot(20, CoH_Data, PCAdf, ax[0])

    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Treatment", numComps=num_comps)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Marker", numComps=num_comps)

    CoH_DF = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    BC_scatter(ax[4], CoH_DF, "pSTAT3", "IL10-50ng")
    BC_scatter(ax[5], CoH_DF, "pSTAT5", "IFNg-50ng")

    return f


def BC_status_plot(compNum, CoH_Data, PCAdf, ax):
    """Plot 5 fold CV by # components"""
    accDF = pd.DataFrame()
    PCA_X = PCAdf.values
    Donor_CoH_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cv = StratifiedKFold(n_splits=5)
    model = LogisticRegression()
    scoresPCA = cross_val_score(model, PCA_X, Donor_CoH_y, cv=cv)
    for i in range(1, compNum + 1):
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
            tFacDF = tFacDF[["Patient 35", "Patient 43", "Patient 44", "Patient 45", "Patient 52", "Patient 54", "Patient 56", "Patient 58", "Patient 63", "Patient 66", "Patient 70", "Patient 79", "Patient 4", "Patient 8", "Patient 406", "Patient 10-T1",  "Patient 10-T2",  "Patient 10-T3", "Patient 15-T1",  "Patient 15-T2",  "Patient 15-T3"]]
            TFAC_X = tFacDF.transpose().values
            model = LogisticRegression()
            scoresTFAC = cross_val_score(model, TFAC_X, Donor_CoH_y, cv=cv)
            accDF = pd.concat([accDF, pd.DataFrame({"Data Type": "Tensor Factorization", "Components": [i], "Accuracy (5-fold CV)": np.mean(scoresTFAC)})])
            accDF = pd.concat([accDF, pd.DataFrame({"Data Type": "All Data", "Components": [i], "Accuracy (5-fold CV)": np.mean(scoresPCA)})])
    accDF = accDF.reset_index(drop=True)
    sns.lineplot(data=accDF, x="Components", y="Accuracy (5-fold CV)", hue="Data Type", ax=ax)
    ax.set(xticks = np.arange(1, compNum + 1))


def BC_scatter(ax, CoH_DF, marker, cytokine):
    """Scatters specific responses"""
    hist_DF = CoH_DF.loc[(CoH_DF.Treatment == cytokine) & (CoH_DF.Marker == marker)]
    hist_DF["Status"] = hist_DF.replace({"Patient": status_dict}).Patient.values
    sns.histplot(data=hist_DF, x="Mean", hue="Status", ax=ax)
    ax.set(title=marker + " in response to " + cytokine, xlabel=marker, ylabel="Count")


status_dict = {"Patient 35": "Healthy", 
                "Patient 43": "Healthy", 
                "Patient 44": "Healthy",
                "Patient 45": "Healthy", 
                "Patient 52": "Healthy",
                "Patient 54": "Healthy",
                "Patient 56": "Healthy",
                "Patient 58": "Healthy", 
                "Patient 63": "Healthy", 
                "Patient 66": "Healthy", 
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
                "Patient 15-T3": "BC"}