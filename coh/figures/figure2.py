"""
This creates Figure 2, tensor factorization of response data.
"""
from os.path import join
import xarray as xa
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .common import subplotLabel, getSetup, path_here
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH
from ..flow import make_CoH_Tensor


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
    tFacAllM, _ = factorTensor(CoH_Data.values, r=num_comps)
    R2Xplot(ax[1], CoH_Data.values, compNum=14)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Patient", cbar=False)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Treatment", cbar=False)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Cell", cbar=False)
    plot_tFac_CoH(ax[5], tFacAllM, CoH_Data, "Marker", cbar=False)

    tc = Decomposition(CoH_Data.to_numpy(), max_rr=12)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=0)

    reduction(ax[6], tc)

    return f
