"""
This creates Figure S7, factorization of receptors.
"""
from .common import subplotLabel, getSetup
from ..tensor import factorTensor, plot_tFac_CoH, varyCompPlots, get_status_rec_df
from ..flow_rec import make_CoH_Tensor_rec


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    CoH_Data = make_CoH_Tensor_rec()
    varyCompPlots([ax[0], ax[4]], 10, CoH_Data, get_status_rec_df())

    tFacAllM, _ = factorTensor(CoH_Data.values, r=4)

    plot_tFac_CoH(ax[1], tFacAllM, CoH_Data, "Patient")
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Cell")
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Marker")

    return f
