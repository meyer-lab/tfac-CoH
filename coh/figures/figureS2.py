"""
This creates Figure S2, factorization of fold-change data.
"""
from ..tensor import (
    factorTensor,
    R2Xplot,
    CoH_LogReg_plot,
    BC_status_plot,
)
from ..flow import make_CoH_Tensor, get_status_df
from .common import subplotLabel, getSetup, plot_tFac_CoH


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (2, 4))

    # Add subplot labels
    subplotLabel(ax)
    CoH_Data = make_CoH_Tensor(just_signal=True, foldChange=True)

    tFacAllM = factorTensor(CoH_Data.to_numpy(), r=8)
    R2Xplot(ax[0], CoH_Data.to_numpy(), compNum=10)
    BC_status_plot(10, CoH_Data, ax[1], get_status_df())
    CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data, get_status_df())
    
    plot_tFac_CoH(ax[3:], tFacAllM, CoH_Data)

    return f
