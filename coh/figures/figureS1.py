"""
This creates Figure 1.
"""
from os.path import join
from .figureCommon import subplotLabel, getSetup
from tensorpack.plot import reduction, tucker_reduction
from tensorpack.tucker import tucker_decomp
from tensorpack import Decomposition
import xarray as xa
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc"))

    # makePCA_df(CoH_Data)
    # plot_PCA(ax[0:2])

    # perform parafac
    decomp_data = CoH_Data.to_numpy()
    nan_index = []
    for i in range(0, decomp_data.shape[0]):
        nan_index.append(~np.isnan(decomp_data[i, :, :, :, :]).any())
    decomp_data = decomp_data[nan_index, :, :, :, :]

    tc = Decomposition(decomp_data, max_rr=12)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    reduction(ax[0], tc)
    tuck = Decomposition(decomp_data, method=tucker_decomp, max_rr=10)
    para = Decomposition(decomp_data, max_rr=10)
    tucker_reduction(ax[1], tuck, para)
    # plot_PCA(ax[0:2])

    return f
