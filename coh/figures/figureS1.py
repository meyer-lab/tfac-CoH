"""
This creates Figure S1, dimensionality reduction size check.
"""
from os.path import join, dirname
import xarray as xa
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .figureCommon import subplotLabel, getSetup


path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((3, 3), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = xa.open_dataarray(join(path_here, "data/CoH_Tensor_DataSet.nc")).to_numpy()

    tc = Decomposition(X, max_rr=8)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    reduction(ax[0], tc)

    return f
