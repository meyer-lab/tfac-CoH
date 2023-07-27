"""
This creates Figure S3, factorization of cell type abundance.
"""
from os.path import join
import xarray as xa
from .common import subplotLabel, getSetup, path_here
from ..tensor import (
    factorTensor,
    R2Xplot,
    plot_tFac_CoH,
    BC_status_plot,
    CoH_LogReg_plot,
)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 6), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    num_comps = 7

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Tensor_Abundance.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, r=num_comps)

    R2Xplot(ax[0], CoH_Data.values, compNum=8)
    BC_status_plot(8, CoH_Data, ax[1])
    CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data, num_comps)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Patient")
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Treatment")
    plot_tFac_CoH(ax[5], tFacAllM, CoH_Data, "Cell")

    return f
