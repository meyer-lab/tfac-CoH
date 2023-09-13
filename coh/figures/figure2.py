"""
This creates Figure 2, tensor factorization of response data.
"""
import pickle
import numpy as np
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .common import subplotLabel, getSetup, plot_tFac_CoH, CoH_Scat_Plot
from ..tensor import factorTensor, BC_status_plot, CoH_LogReg_plot
from ..flow import make_CoH_Tensor, get_status_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (4, 3), multz={0: 2})

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    CoH_Data = make_CoH_Tensor(just_signal=True)

    with open('./coh/data/signaling.pkl', 'rb') as ff:
        tFacAllM = pickle.load(ff) # 12 component

    plot_tFac_CoH(ax[2:], tFacAllM, CoH_Data)

    tc = Decomposition(CoH_Data.to_numpy(), max_rr=14, method=factorTensor)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=0)

    reduction(ax[6], tc)

    # R2X plot
    ax[1].scatter(np.arange(1, len(tc.TR2X) + 1), tc.TR2X, c="k", s=20.0)
    ax[1].set(
        title="R2X",
        ylabel="Variance Explained",
        xlabel="Number of Components",
        ylim=(0, 1),
        xlim=(0, len(tc.TR2X) + 0.5),
        xticks=np.arange(0, len(tc.TR2X) + 1),
    )

    CoH_Data = make_CoH_Tensor(just_signal=True)

    with open('./coh/data/signaling.pkl', 'rb') as ff:
        tFacAllM = pickle.load(ff) # 12 component

    BC_status_plot(13, CoH_Data, ax[7], get_status_df())
    CoH_LogReg_plot(ax[8], tFacAllM, CoH_Data, get_status_df())
    CoH_Scat_Plot(ax[9], tFacAllM, CoH_Data, "Patient", plot_comps=[5, 10], status_df=get_status_df())

    return f
