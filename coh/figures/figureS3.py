"""This creates Figure S3, factorization of fold-change data."""

from ..flow import get_status_df, make_CoH_Tensor
from ..tensor import (
    BC_status_plot,
    CoH_LogReg_plot,
    R2Xplot,
    factorTensor,
)
from .common import getSetup, plot_tFac_CoH, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 3), (2, 4))

    # Add subplot labels
    subplotLabel(ax)
    CoH_Data = make_CoH_Tensor(just_signal=True, foldChange=True)

    tFacAllM = factorTensor(CoH_Data.to_numpy(), r=8)
    R2Xplot(ax[0], CoH_Data.to_numpy(), compNum=10)
    BC_status_plot(10, CoH_Data, ax[1], get_status_df())
    CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data, get_status_df())

    plot_tFac_CoH(ax[3:], tFacAllM, CoH_Data)

    return f
