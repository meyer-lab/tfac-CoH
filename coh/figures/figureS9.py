"""
This creates Figure S8, signaling R2X and decomposition.
"""
import pickle
import numpy as np
from tensorpack.plot import reduction
from tensorpack import Decomposition
from .common import subplotLabel, getSetup
from ..tensor import factorTensor
from ..flow_rec import make_CoH_Tensor_rec


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 3), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    CoH_Data = make_CoH_Tensor_rec()

    tc = Decomposition(CoH_Data.to_numpy(), max_rr=8, method=factorTensor)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=0)

    # R2X plot
    ax[0].scatter(np.arange(1, len(tc.TR2X) + 1), tc.TR2X, c="k", s=20.0)

    ax[0].set(
        title="R2X",
        ylabel="Variance Explained",
        xlabel="Number of Components",
        ylim=(0, 1),
        xlim=(0, len(tc.TR2X) + 0.5),
        xticks=np.arange(0, len(tc.TR2X) + 1),
    )

    reduction(ax[1], tc)

    return f
