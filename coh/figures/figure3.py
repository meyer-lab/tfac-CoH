"""
This creates Figure 3, tensor factorization of receptor data.
"""
import xarray as xa
import numpy as np
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH


path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    num_comps = 4
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Rec.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    R2Xplot(ax[1], CoH_Data.values, compNum=8)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Patient", numComps=num_comps, cbar=False, rec=True)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Cell", numComps=num_comps, cbar=False)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Marker", numComps=num_comps, cbar=False)

    tc = Decomposition(CoH_Data, max_rr=8)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    reduction(ax[5], tc)

    return f
