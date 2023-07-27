"""
This creates Figure 3, tensor factorization of receptor data.
"""
import xarray as xa
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .common import subplotLabel, getSetup, path_here
from os.path import join
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    num_comps = 4
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Rec.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, r=num_comps)
    R2Xplot(ax[1], CoH_Data.values, compNum=8)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Patient", cbar=False, rec=True)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Cell", cbar=False)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Marker", cbar=False)

    tc = Decomposition(CoH_Data.to_numpy(), max_rr=8)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=0)

    reduction(ax[5], tc)

    return f
