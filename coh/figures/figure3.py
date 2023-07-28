"""
This creates Figure 3, tensor factorization of receptor data.
"""
import xarray as xa
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .common import subplotLabel, getSetup
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH
from ..flow_rec import make_CoH_Tensor_rec


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    CoH_Data = make_CoH_Tensor_rec()
    tFacAllM, _ = factorTensor(CoH_Data.values, r=5)
    R2Xplot(ax[1], CoH_Data.values, compNum=8)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Patient", cbar=False, rec=True)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Cell", cbar=False)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Marker", cbar=False)

    tc = Decomposition(CoH_Data.to_numpy(), max_rr=8)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=0)

    reduction(ax[5], tc)

    return f
