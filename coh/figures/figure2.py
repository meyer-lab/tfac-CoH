"""
This creates Figure 1.
"""
import xarray as xa
import os
import warnings
warnings.filterwarnings("ignore")
from tensorpack import Decomposition
from tensorpack.tucker import tucker_decomp
from tensorpack.plot import reduction, tucker_reduction
from .figureCommon import subplotLabel, getSetup
from os.path import join
from ..tensor import make_alldata_DF

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc"))
    
    #makePCA_df(CoH_Data)
    #plot_PCA(ax[0:2])
    
    # perform parafac
    tc = Decomposition(CoH_Data.to_numpy()[0:-15, :, :, :, :], max_rr=12)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    reduction(ax[0], tc)
    tuck = Decomposition(CoH_Data.to_numpy()[0:-15, :, :, :, :], method=tucker_decomp, max_rr=10)
    para = Decomposition(CoH_Data.to_numpy()[0:-15, :, :, :, :], max_rr=10)
    tucker_reduction(ax[1], tuck, para)
    #plot_PCA(ax[0:2])

    return f
