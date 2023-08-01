"""
This creates Figure 4, classification analysis.
"""
import seaborn as sns
import pandas as pd
from .common import subplotLabel, getSetup
from ..tensor import factorTensor, CoH_LogReg_plot, BC_status_plot, get_status_df
from ..flow_rec import make_CoH_Tensor_rec
from ..flow import make_CoH_Tensor


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 6), (3, 4), multz={0: 3})

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    CoH_Data = make_CoH_Tensor(just_signal=True)
    tFacAllM = factorTensor(CoH_Data.values, r=12)

    BC_status_plot(13, CoH_Data, ax[1])
    CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data)
    CoH_Scat_Plot(ax[3], tFacAllM, CoH_Data, "Patient", plot_comps=[2, 9])

    CoH_Data_R = make_CoH_Tensor_rec()
    tFacAllM_R = factorTensor(CoH_Data_R.values, r=3)

    BC_status_plot(6, CoH_Data_R, ax[5], rec=True)
    CoH_LogReg_plot(ax[6], tFacAllM_R, CoH_Data_R)
    CoH_Scat_Plot(ax[7], tFacAllM_R, CoH_Data_R, "Patient", plot_comps=[1, 2])

    return f


def CoH_Scat_Plot(ax, tFac, CoH_Array, mode, plot_comps):
    """Plots bar plot for spec"""
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame()

    for i in range(0, mode_facs.shape[1]):
        tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, i], "Component": (i + 1), mode: mode_labels})])
    tFacDF = tFacDF.loc[tFacDF["Component"].isin(plot_comps)]
    tFacDF = tFacDF.pivot(index=mode, columns='Component', values='Component_Val')
    if mode == "Patient":
        status_df = get_status_df().set_index("Patient")
        tFacDF = pd.concat([tFacDF, status_df], axis=1)
        sns.scatterplot(data=tFacDF, x=plot_comps[0], y=plot_comps[1], hue="Status", style="Status", ax=ax)
    else:
        sns.scatterplot(data=tFacDF, x=plot_comps[0], y=plot_comps[1], ax=ax)
    ax.set(xlabel="Component " + str(plot_comps[0]), ylabel="Component " + str(plot_comps[1]))
