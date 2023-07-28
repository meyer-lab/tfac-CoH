"""
This creates Figure 2, tensor factorization of response data.
"""
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .common import subplotLabel, getSetup
from ..tensor import factorTensor, plot_tFac_CoH, varyCompPlots, get_status_df
from ..flow import make_CoH_Tensor


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 9), (3, 3), multz={0: 2})

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")
    CoH_Data = make_CoH_Tensor(just_signal=True, foldChange=False)

    tFacAllM, _ = factorTensor(CoH_Data.values, r=12)

    varyCompPlots([ax[1], ax[0]], 3, CoH_Data, get_status_df())

    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Patient")
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Treatment")
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Cell")
    plot_tFac_CoH(ax[5], tFacAllM, CoH_Data, "Marker")

    tc = Decomposition(CoH_Data.to_numpy(), max_rr=12)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=0)

    reduction(ax[6], tc)

    return f
