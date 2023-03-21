"""
This creates Figure 2, tensor factorization of response data.
"""
import xarray as xa
import numpy as np
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH
from ..flow import make_CoH_Tensor


path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 9), (3, 3), multz={0: 2})

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")
    make_CoH_Tensor(just_signal=True)

    num_comps = 12
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Tensor_DataSet.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    R2Xplot(ax[1], CoH_Data.values, compNum=15)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Patient", numComps=num_comps, cbar=False)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Treatment", numComps=num_comps, cbar=False)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Cell", numComps=num_comps, cbar=False)
    plot_tFac_CoH(ax[5], tFacAllM, CoH_Data, "Marker", numComps=num_comps, cbar=False)

    tc = Decomposition(CoH_Data.to_numpy(), max_rr=15)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    reduction(ax[6], tc)

    return f
