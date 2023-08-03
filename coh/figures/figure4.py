"""
This creates Figure 4, classification analysis.
"""
import pickle
import seaborn as sns
import pandas as pd
from .common import subplotLabel, getSetup
from ..tensor import factorTensor, CoH_LogReg_plot, BC_status_plot
from ..flow_rec import make_CoH_Tensor_rec, get_status_rec_df
from ..flow import make_CoH_Tensor, get_status_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 6), (3, 4), multz={0: 3})

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    CoH_Data = make_CoH_Tensor(just_signal=True)

    with open('./coh/data/signaling.pkl', 'rb') as ff:
        tFacAllM = pickle.load(ff) # 12 component

    BC_status_plot(13, CoH_Data, ax[1], get_status_df())
    CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data, get_status_df())
    CoH_Scat_Plot(ax[3], tFacAllM, CoH_Data, "Patient", plot_comps=[5, 10], status_df=get_status_df())

    CoH_Data_R = make_CoH_Tensor_rec()
    tFacAllM_R = factorTensor(CoH_Data_R.to_numpy(), r=5)

    BC_status_plot(6, CoH_Data_R, ax[5], get_status_rec_df())
    CoH_LogReg_plot(ax[6], tFacAllM_R, CoH_Data_R, get_status_rec_df())
    CoH_Scat_Plot(ax[7], tFacAllM_R, CoH_Data_R, "Patient", plot_comps=[1, 2], status_df=get_status_rec_df())

    return f


def CoH_Scat_Plot(ax, tFac, CoH_Array, mode, plot_comps, status_df):
    """Plots bar plot for spec"""
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame(mode_facs, index=mode_labels, columns=[i + 1 for i in range(mode_facs.shape[1])])

    if mode == "Patient":
        tFacDF = pd.concat([tFacDF, status_df.set_index("Patient")], axis=1)
        sns.scatterplot(data=tFacDF, x=plot_comps[0], y=plot_comps[1], hue="Status", style="Status", ax=ax)
    else:
        sns.scatterplot(data=tFacDF, x=plot_comps[0], y=plot_comps[1], ax=ax)
    ax.set(xlabel="Component " + str(plot_comps[0]), ylabel="Component " + str(plot_comps[1]))
