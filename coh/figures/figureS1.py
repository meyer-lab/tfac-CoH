"""
This creates Figure 1.
"""
from os.path import join, dirname
import xarray as xa
import matplotlib.pyplot as plt
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .figureCommon import subplotLabel, getSetup


path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc")).to_numpy()

    tc = Decomposition(X, max_rr=8)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    reduction(ax[0], tc)

    return f
