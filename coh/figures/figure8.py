"""
This creates Figure 4.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as plt
from scipy.stats import zscore
from os.path import join, dirname
from ..flow import make_CoH_Tensor, get_status_df
from .common import subplotLabel, getSetup


plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 7), (1, 1))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df(foldChange=True)
    # make_CoH_Tensor(just_signal=True, foldChange=True)

    #make_alldata_DF(CoH_Data, PCA=False, foldChange=True)
    CoH_DF = pd.read_csv("./coh/data/CoH_Flow_DF.csv", index_col=0)
    treatments = np.array(["IL2-50ng", "IL4-50ng", "IL6-50ng", "IL10-50ng", "IFNg-50ng", "TGFB-50ng", "IFNg-50ng+IL6-50ng", "Untreated"])
    markers = np.array(["pSTAT1", "pSTAT3", "pSTAT4", "pSTAT5", "pSTAT6", "pSmad1-2"])
    CoH_DF = CoH_DF.loc[(CoH_DF.Treatment.isin(treatments)) & (CoH_DF.Marker.isin(markers))].dropna()
    CoH_DF_R = pd.read_csv("./coh/data/CoH_Rec_DF.csv", index_col=0)
    CoH_DF_R["Treatment"] = "Receptor"

    CoH_DF = pd.concat([CoH_DF, CoH_DF_R])
    print(CoH_DF)
    
    meanDF = CoH_DF.groupby(["Patient", "Cell", "Treatment", "Marker"]).mean().reset_index()
    meanDF = (meanDF.pivot(index=["Patient", "Cell", "Treatment"], columns="Marker", values="Mean").reset_index().set_index("Patient"))
    meanDF.iloc[:, 2::] = meanDF.iloc[:, 2::].apply(zscore)
    print(meanDF)
   

    return f


def get_jointDF(CoH_DF, CoH_DF_R):
    """Plots possible correlation of dysregulation"""
    CoH_DF = CoH_DF.loc[CoH_DF.Patient != "Patient 406"]
    CoH_DF_R = CoH_DF_R.loc[CoH_DF_R.Patient != "Patient 19186-12"]

    CoH_DF = CoH_DF.groupby(["Cell", "Patient", "Treatment", "Marker"]).Mean.mean().reset_index()
    CoH_DF_R = CoH_DF_R.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()

    jointDF = pd.DataFrame()
    jointDF = CoH_DF.set_index("Patient").join(
        CoH_DF_R.set_index("Patient"), on="Patient"
    )
    jointDF.iloc[:, 2::] = jointDF.iloc[:, 2::].apply(zscore)

    status_DF = get_status_df()
    jointDF = jointDF.set_index("Patient").join(status_DF.set_index("Patient"), on="Patient")
    
    return jointDF
