"""
This creates Figure 8, an examination of tucker decomposition for response data.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from .common import subplotLabel, getSetup

from ..tensor import factorTensor
from ..flow_rec import make_CoH_Tensor_rec
from ..flow import make_CoH_Tensor
from tensordata.zohar import data

from tensorpack.tucker import tucker_decomp
from tensorpack.plot import tucker_reduced_Dsize, tucker_reduction

from tensorpack.decomposition import Decomposition
    
def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 5), (1, 1))

    # Add subplot labels
    subplotLabel(ax)
    CoH_Data = make_CoH_Tensor(just_signal=True)
    print(CoH_Data)
    CoH_Data = np.nan_to_num(CoH_Data)
    
    tuck_decomp = Decomposition(CoH_Data, method=tucker_decomp)
    cp_decomp = Decomposition(CoH_Data, max_rr=15)

    tucker_reduction(ax[0], tuck_decomp, cp_decomp)

    return f
