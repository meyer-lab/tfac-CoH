"""This creates Figure S10, jackknife FMS."""

import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import seaborn as sns
from ..flow import make_CoH_Tensor, get_status_dict
from ..flow_rec import make_CoH_Tensor_rec, get_status_dict_rec
from .common import getSetup, subplotLabel
from ..tensor import factorTensor
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms
from tqdm import tqdm


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 3), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    data_sig = make_CoH_Tensor(just_signal=True)

    with open("./coh/data/signaling.pkl", "rb") as ff:
        tFac_sig = pickle.load(ff)  # 12 component

    data_rec = make_CoH_Tensor_rec()
    tFac_rec = factorTensor(data_rec.to_numpy(), r=5)

    jackknife_plot(data_sig, tFac_sig, sample_axis=0, rank=12, patient_dict=get_status_dict(), ax=ax[0])
    jackknife_plot(data_rec, tFac_rec, sample_axis=0, rank=5, patient_dict=get_status_dict_rec(), ax=ax[1])

    return f

def jackknife_plot(data, tfac_orig, sample_axis, rank, patient_dict, ax):
    """Plots jacknife swarm plot upon leaving one slice of tensor out per run"""
    fms_scores = np.zeros(data.shape[sample_axis])
    statuses = np.empty(data.shape[sample_axis], dtype="<U10")
    for i in tqdm(range(data.shape[sample_axis])):
        tensor_j = np.delete(data.to_numpy(), i, sample_axis)
        cp_j = factorTensor(tensor_j, rank)
        fms_scores[i] = fms(tfac_orig, cp_j, consider_weights=True, skip_mode=sample_axis)
        statuses[i] = patient_dict[data.Patient.to_numpy()[i]]

    fmsDF = pd.DataFrame({"FMS Scores": fms_scores, "Status": statuses})
    sns.swarmplot(data=fmsDF, y="FMS Scores", hue="Status", ax=ax)
    ax.set(ylim=(0.5, 1))
