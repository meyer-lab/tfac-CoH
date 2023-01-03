"""
This creates Figure 1.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import cp_normalize, perform_CP
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..flow import make_flow_df, make_CoH_Tensor, make_CoH_Tensor_abund
from ..tensor import factorTensor, R2Xplot, plot_tFac_CoH

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 9), (3, 3), multz={0: 2})

    # Add subplot labels
    subplotLabel(ax)
    ax[0].axis("off")
    # make_flow_df(subtract=False, abundance=True, foldChange=False)
    # make_CoH_Tensor_abund()

    num_comps = 12
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc"))

    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    cp_normalize(tFacAllM)
    R2Xplot(ax[1], CoH_Data.values, compNum=15)
    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Patient", numComps=num_comps, cbar=False)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Time", numComps=num_comps, cbar=False)
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Treatment", numComps=num_comps, cbar=False)
    plot_tFac_CoH(ax[5], tFacAllM, CoH_Data, "Cell", numComps=num_comps, cbar=False)
    plot_tFac_CoH(ax[6], tFacAllM, CoH_Data, "Marker", numComps=num_comps, cbar=False)

    return f
