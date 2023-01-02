"""
This creates Figure S2. NN Factorization.
"""
import xarray as xa
import os
from tensorpack.cmtf import cp_normalize
from .figureCommon import subplotLabel, getSetup
from os.path import join
from ..flow import make_flow_df, make_CoH_Tensor
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH, CoH_LogReg_plot

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((16, 12), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    #make_flow_df(subtract=False, abundance=False)
    # make_CoH_Tensor(subtract=False)

    num_comps = 12

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Tensor_DataSet_FC.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    cp_normalize(tFacAllM)
    CoH_LogReg_plot(ax[0], tFacAllM, CoH_Data, num_comps)
    #R2Xplot(ax[0], CoH_Data.values, compNum=8)
    plot_tFac_CoH(ax[1], tFacAllM, CoH_Data, "Patient", numComps=num_comps, nn=False)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Time", numComps=num_comps, nn=False)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Treatment", numComps=num_comps, nn=False)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Cell", numComps=num_comps, nn=False)
    plot_tFac_CoH(ax[5], tFacAllM, CoH_Data, "Marker", numComps=num_comps, nn=False)

    return f
