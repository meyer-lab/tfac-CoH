"""
This creates Figure S3, factorization of cell type abundance.
"""
from .common import subplotLabel, getSetup, plot_tFac_CoH
from ..tensor import (
    factorTensor,
    R2Xplot,
    BC_status_plot,
    CoH_LogReg_plot,
)
from ..flow import make_CoH_Tensor_abund


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 6), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    CoH_Data = make_CoH_Tensor_abund()
    tFacAllM = factorTensor(CoH_Data.values, r=7)

    R2Xplot(ax[0], CoH_Data.values, compNum=8)
    BC_status_plot(8, CoH_Data, ax[1])
    CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data)

    plot_tFac_CoH(ax[3:], tFacAllM, CoH_Data)

    return f
