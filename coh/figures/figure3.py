"""
This creates Figure 1.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from tensorpack.cmtf import cp_normalize
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..tensor import factorTensor, CoH_LogReg_plot, plot_tFac_CoH, make_alldata_DF, BC_status_plot, BC_scatter
from ..flow import make_flow_df


path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 6), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df()
    # make_CoH_Tensor(just_signal=True)

    num_comps = 12
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    cp_normalize(tFacAllM)
    make_alldata_DF(CoH_Data, PCA=False)
    CoH_LogReg_plot(ax[1], tFacAllM, CoH_Data, num_comps)
    matrix_DF = pd.read_csv(join(path_here, "data/CoH_Matrix.csv"), index_col=0).dropna(axis='columns').set_index("Patient")

    BC_status_plot(10, CoH_Data, matrix_DF, ax[0], abund=False)

    plot_tFac_CoH(ax[2], tFacAllM, CoH_Data, "Treatment", numComps=num_comps)
    plot_tFac_CoH(ax[3], tFacAllM, CoH_Data, "Cell", numComps=num_comps)
    CoH_DF = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    BC_scatter(ax[4], CoH_DF, "pSTAT3", "IL10-50ng")
    BC_scatter(ax[5], CoH_DF, "pSTAT5", "IL2-50ng", cells=["Treg", "CD4+", "CD8+"])

    return f
