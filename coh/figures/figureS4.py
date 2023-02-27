"""
This creates Figure 1.
"""
import xarray as xa
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df(subtract=False, abundance=True)
    # make_CoH_Tensor_abund()

    num_comps = 6

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Tensor_Abundance.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    R2Xplot(ax[0], CoH_Data.values, compNum=10)
    plot_tFac_CoH(ax[1], tFacAllM, CoH_Data, "Patient", numComps=num_comps)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Time", numComps=num_comps)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Treatment", numComps=num_comps)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Cell", numComps=num_comps)

    return f
