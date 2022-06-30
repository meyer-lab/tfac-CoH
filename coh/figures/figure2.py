"""
This creates Figure 1.
"""
import xarray as xa
from tensorpack import Decomposition
from tensorpack.tucker import tucker_decomp
from tensorpack.plot import reduction, tucker_reduction
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc"))
    # perform parafac
    tc = Decomposition(CoH_Data.to_numpy(), max_rr=8)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=0)

    reduction(ax[0], tc)
    #tuck = Decomposition(CoH_Data.to_numpy(), method=tucker_decomp, max_rr=2)
    #para = Decomposition(CoH_Data.to_numpy())
    #tucker_reduction(ax[1], tuck, para)

    return f