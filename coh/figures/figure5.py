"""
This creates Figure 5.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
from scipy.stats import ttest_ind
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import cp_normalize, perform_CP
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..flow_rec import make_flow_df_rec, make_CoH_Tensor_rec
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH, CoH_LogReg_plot, BC_status_plot_rec, get_status_dict_rec

plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 9), (4, 3))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df_rec()
    # make_CoH_Tensor_rec()

    num_comps = 8

    CoH_Data_R = xa.open_dataarray(join(path_here, "data/CoH_Rec.nc"))
    #make_allrec_DF(CoH_Data_R)
    matrix_DF_R = pd.read_csv(join(path_here, "data/CoH_Matrix_Rec.csv"), index_col=0).dropna(axis='columns').set_index("Patient")
    #BC_status_plot_rec(20, CoH_Data_R, matrix_DF_R, ax[1])
    CoH_Data_DF = pd.read_csv(join(path_here, "data/CoH_Rec_DF.csv"))
    for i, rec in enumerate(np.array(["IFNg R1", "TGFB RII", "PD1", "PD_L1", "IL2Ra", "IL2RB", "IL4Ra", "IL6Ra", "IL6RB", "IL7Ra", "IL10R", "IL12RI"])):
        print(rec)
        BC_scatter_cells_rec(ax[i], CoH_Data_DF, rec, filter=False)

    return f


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
    PCAdf.to_csv(path_here + "/data/CoH_Matrix_Rec.csv")


def BC_scatter_cells_rec(ax, CoH_DF, marker, filter=False):
    """Scatters specific responses"""
    status_dict = get_status_dict_rec()
    hist_DF = CoH_DF.loc[(CoH_DF.Marker == marker)]
    hist_DF = hist_DF.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()
    hist_DF["Status"] = hist_DF.replace({"Patient": status_dict}).Patient.values


    filt_cells = []
    pvals = []
    for cell in hist_DF.Cell.unique():
        BC_samps = hist_DF.loc[(hist_DF.Status == "BC") & (hist_DF.Cell == cell)].Mean.values
        H_samps = hist_DF.loc[(hist_DF.Status == "Healthy") & (hist_DF.Cell == cell)].Mean.values
        t_res = ttest_ind(BC_samps, H_samps)
        if t_res[1] < (0.05 / hist_DF.Cell.unique().size):
            filt_cells.append(cell)
            if t_res[1] * hist_DF.Cell.unique().size < 0.0005:
                pvals.append("***")
            elif t_res[1] * hist_DF.Cell.unique().size < 0.005:
                pvals.append("**")
            elif t_res[1] * hist_DF.Cell.unique().size < 0.05:
                pvals.append("*")
            else:
                pvals.append("****")
        else:
            if not filter:
                pvals.append("ns")
    if filter:
        hist_DF = hist_DF.loc[hist_DF.Cell.isin(filt_cells)]
    sns.boxplot(data=hist_DF, y="Mean", x="Cell", hue="Status", ax=ax)
    ax.set(title=marker, ylabel=marker, xlabel="Status")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    boxpairs = []
    for cell in hist_DF.Cell.unique():
        boxpairs.append([(cell, "Healthy"), (cell, "BC")])
    if filter:
        add_stat_annotation(ax=ax, data=hist_DF, x="Cell", y="Mean", hue="Status", box_pairs=boxpairs, text_annot_custom=pvals, perform_stat_test=False, loc='inside', pvalues=np.tile(0, len(filt_cells)), verbose=0)
    else:
        add_stat_annotation(ax=ax, data=hist_DF, x="Cell", y="Mean", hue="Status", box_pairs=boxpairs, text_annot_custom=pvals, perform_stat_test=False, loc='inside', pvalues=np.tile(0, len(hist_DF.Cell.unique())), verbose=0)
