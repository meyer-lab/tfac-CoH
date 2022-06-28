"""
This creates Figure 1.
"""
import xarray as xa
import tensorly as tl
import numpy as np
import seaborn as sns
import pandas as pd
import os
from tensorpack import Decomposition
from tensorpack.tucker import tucker_decomp
from tensorpack.plot import reduction, tucker_reduction
from .figureCommon import subplotLabel, getSetup
from os.path import join

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    CoH_Data = xa.open_dataarray(join(path_here, "data/NN CoH Tensor DataSet.nc"))
    # perform parafac
    tc = Decomposition(CoH_Data.to_numpy(), max_rr=12)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    reduction(ax[0], tc)
    tuck = Decomposition(CoH_Data.to_numpy(), method=tucker_decomp)
    para = Decomposition(CoH_Data.to_numpy())
    tucker_reduction(ax[1], tuck, para)

    return f