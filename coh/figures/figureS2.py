"""
This creates Figure S2, factorization of fold-change data.
"""
from ..tensor import (
    factorTensor,
    R2Xplot,
    CoH_LogReg_plot,
    BC_status_plot,
)
from ..flow import make_CoH_Tensor
from .common import subplotLabel, getSetup, plot_tFac_CoH


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (2, 4))

    # Add subplot labels
    subplotLabel(ax)
    CoH_Data = make_CoH_Tensor(just_signal=True, foldChange=True)

    tFacAllM = factorTensor(CoH_Data.values, r=8)
    R2Xplot(ax[0], CoH_Data.values, compNum=10)
    BC_status_plot(10, CoH_Data, ax[1])
    CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data)
    
    plot_tFac_CoH(ax[3:], tFacAllM, CoH_Data)

    return f
