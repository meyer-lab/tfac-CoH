"""
This creates Figure S2, factorization of fold-change data.
"""
from ..tensor import (
    factorTensor,
    R2Xplot,
    plot_tFac_CoH,
    CoH_LogReg_plot,
    BC_status_plot,
)
from ..flow import make_CoH_Tensor
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (2, 4))

    # Add subplot labels
    subplotLabel(ax)
    CoH_Data = make_CoH_Tensor(just_signal=True, foldChange=True)

    tFacAllM, _ = factorTensor(CoH_Data.values, r=8)
    R2Xplot(ax[0], CoH_Data.values, compNum=10)
    BC_status_plot(10, CoH_Data, ax[1])
    CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Patient")
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Treatment")
    plot_tFac_CoH(ax[5], tFacAllM, CoH_Data, "Cell")
    plot_tFac_CoH(ax[6], tFacAllM, CoH_Data, "Marker")

    return f
