"""
This creates Figure 5, heatmap (clustered factor correlations).
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup

from ..tensor import factorTensor, get_status_df
from ..flow_rec import make_CoH_Tensor_rec
from ..flow import make_CoH_Tensor


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 1))

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    num_comps = 12
    CoH_Data = make_CoH_Tensor(just_signal=True)
    tFacAllM = factorTensor(CoH_Data.values, r=num_comps)

    num_comps = 4
    CoH_Data_R = make_CoH_Tensor_rec()
    tFacAllM_R = factorTensor(CoH_Data_R.values, r=num_comps)

    f = CoH_Factor_HM(ax[0], tFacAllM, CoH_Data, tFacAllM_R, CoH_Data_R, sig_comps=[2, 5, 9, 10], rec_comps=[1, 2, 4])

    return f


def CoH_Factor_HM(ax, tFac, CoH_Array, tFac_R, CoH_Array_R, sig_comps, rec_comps,):
    """Plots bar plot for spec"""
    mode_labels = CoH_Array["Patient"]
    coord = CoH_Array.dims.index("Patient")
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame()

    status_DF = get_status_df()
    BC_Patients = status_DF.loc[status_DF.Status == "BC"].Patient.unique()

    for i in sig_comps:
        tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, i - 1], "Signaling Component": "Sig. Comp " + str(i), "Patient": mode_labels})])
    tFacDF = tFacDF.loc[tFacDF.Patient.isin(BC_Patients)]
    tFacDF = tFacDF.pivot(index="Patient", columns='Signaling Component', values='Component_Val')

    mode_labels_R = CoH_Array_R["Patient"]
    coord_R = CoH_Array_R.dims.index("Patient")
    mode_facs_R = tFac_R[1][coord_R]
    tFacDF_R = pd.DataFrame()

    for i in rec_comps:
        tFacDF_R = pd.concat([tFacDF_R, pd.DataFrame({"Component_Val": mode_facs_R[:, i - 1], "Receptor Component": "Rec. Comp " + str(i), "Patient": mode_labels_R})])
    tFacDF_R = tFacDF_R.loc[tFacDF_R.Patient.isin(BC_Patients)]
    tFacDF_R = tFacDF_R.pivot(index="Patient", columns='Receptor Component', values='Component_Val')
    tFacDF = tFacDF.drop(index="Patient 406")

    plot_DF = tFacDF.join(tFacDF_R, on="Patient")

    cov_DF = plot_DF.cov()
    Vi = np.linalg.pinv(cov_DF, hermitian=True)  # Inverse covariance matrix
    Vi_diag = Vi.diagonal()
    D = np.diag(np.sqrt(1 / Vi_diag))
    pCor = -1 * (D @ Vi @ D)  # Partial correlation matrix
    pCor[np.diag_indices_from(pCor)] = 1
    pCorr_DF = pd.DataFrame(pCor, columns=cov_DF.columns, index=cov_DF.columns)

    cmap = sns.color_palette("vlag", as_cmap=True)
    f = sns.clustermap(data=pCorr_DF, robust=True, vmin=-1, vmax=1, row_cluster=True, col_cluster=True, annot=True, cmap=cmap, figsize=(8, 8))
    return f
