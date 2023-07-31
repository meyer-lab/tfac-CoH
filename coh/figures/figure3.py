"""
This creates Figure 3, tensor factorization of receptor data.
"""
import numpy as np
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .common import subplotLabel, getSetup, plot_tFac_CoH
from ..tensor import factorTensor
from ..flow_rec import make_CoH_Tensor_rec


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")

    CoH_Data = make_CoH_Tensor_rec()
    tFacAllM = factorTensor(CoH_Data.values, r=5)
    
    plot_tFac_CoH(ax[2:], tFacAllM, CoH_Data)

    tc = Decomposition(CoH_Data.to_numpy(), max_rr=8)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=0)

    reduction(ax[5], tc)

    # R2X plot
    ax[1].scatter(np.arange(1, len(tc.TR2X) + 1), tc.TR2X, c="k", s=20.0)
    ax[1].set(
        title="R2X",
        ylabel="Variance Explained",
        xlabel="Number of Components",
        ylim=(0, 1),
        xlim=(0, len(tc.TR2X) + 0.5),
        xticks=np.arange(0, len(tc.TR2X) + 1),
    )

    return f
