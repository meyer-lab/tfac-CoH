"""
This creates Figure S2, factorization of fold-change data.
"""
from os.path import join
import xarray as xa
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH, CoH_LogReg_plot, BC_status_plot
from .common import subplotLabel, getSetup, path_here


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (2, 4))

    # Add subplot labels
    subplotLabel(ax)
    num_comps = 8
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Tensor_DataSet_FC.nc"))

    tFacAllM, _ = factorTensor(CoH_Data.values, r=num_comps)
    R2Xplot(ax[0], CoH_Data.values, compNum=10)
    BC_status_plot(10, CoH_Data, ax[1])
    CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data, num_comps)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Patient")
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Treatment")
    plot_tFac_CoH(ax[5], tFacAllM, CoH_Data, "Cell")
    plot_tFac_CoH(ax[6], tFacAllM, CoH_Data, "Marker")

    return f
