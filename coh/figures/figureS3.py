"""
This creates Figure S2. NN Factorization.
"""
import xarray as xa
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH, CoH_LogReg_plot

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 12), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    num_comps = 12

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Tensor_DataSet_FC.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    CoH_LogReg_plot(ax[0], tFacAllM, CoH_Data, num_comps)
    R2Xplot(ax[0], CoH_Data.values, compNum=8)
    plot_tFac_CoH(ax[1], tFacAllM, CoH_Data, "Patient", numComps=num_comps, nn=False)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Treatment", numComps=num_comps, nn=False)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Cell", numComps=num_comps, nn=False)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Marker", numComps=num_comps, nn=False)

    return f
