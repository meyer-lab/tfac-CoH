"""
This creates Figure S7, factorization of receptors.
"""
import xarray as xa
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..tensor import factorTensor, plot_tFac_CoH, CoH_LogReg_plot

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df_rec()
    # make_CoH_Tensor_rec()

    num_comps = 4

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Rec.nc"))
    #make_alldata_DF(CoH_Data, PCA=False, basal=True)
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    #R2Xplot(ax[0], CoH_Data.values, compNum=10)
    plot_tFac_CoH(ax[1], tFacAllM, CoH_Data, "Patient", numComps=num_comps, rec=True)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Cell", numComps=num_comps)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Marker", numComps=num_comps)
    CoH_LogReg_plot(ax[4], tFacAllM, CoH_Data, num_comps)

    return f
