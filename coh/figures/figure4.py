"""
This creates Figure 4.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import cp_normalize, perform_CP
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..flow_rec import make_flow_df_rec, make_CoH_Tensor_rec
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    #make_flow_df_rec()
    #make_CoH_Tensor_rec()

    num_comps = 1

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoH_Rec.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    cp_normalize(tFacAllM)
    R2Xplot(ax[0], CoH_Data.values, compNum=5)
    plot_tFac_CoH(ax[1], tFacAllM, CoH_Data, "Patient", numComps=num_comps, rec=True)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Cell", numComps=num_comps)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Marker", numComps=num_comps)


    return f
