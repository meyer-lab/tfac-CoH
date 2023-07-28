"""
This creates Figure S3, factorization of cell type abundance.
"""
from .common import subplotLabel, getSetup
from ..tensor import (
    factorTensor,
    plot_tFac_CoH,
    CoH_LogReg_plot,
    varyCompPlots,
    get_status_df
)
from ..flow import make_CoH_Tensor_abund


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 6), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    CoH_Data = make_CoH_Tensor_abund()
    tFacAllM, _ = factorTensor(CoH_Data.values, r=7)

    varyCompPlots([ax[0], ax[2]], 8, CoH_Data, get_status_df())

    CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Patient")
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Treatment")
    plot_tFac_CoH(ax[5], tFacAllM, CoH_Data, "Cell")

    return f
