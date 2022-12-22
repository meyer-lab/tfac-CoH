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
    ax, f = getSetup((8, 6), (3, 2), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df()
    # make_CoH_Tensor(just_signal=True)

    num_comps = 12
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)
    cp_normalize(tFacAllM)
    #make_alldata_DF(CoH_Data, PCA=False)
   
    
    matrix_DF = pd.read_csv(join(path_here, "data/CoH_Matrix.csv"), index_col=0).dropna(axis='columns').set_index("Patient")

    #BC_status_plot(15, CoH_Data, matrix_DF, ax[1], abund=False)
    #CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data, num_comps)

    CoH_Bar_Plot(ax[3], tFacAllM, CoH_Data, "Treatment", numComps=12, plot_comps=[8, 11])
    CoH_Bar_Plot(ax[4], tFacAllM, CoH_Data, "Cell", numComps=12, plot_comps=[8, 11])
    #CoH_DF = pd.read_csv(join(path_here, "data/CoH_Flow_DF.csv"))
    #BC_scatter(ax[5], CoH_DF, "pSTAT3", "IL10-50ng")
    #BC_scatter(ax[6], CoH_DF, "pSTAT5", "IL2-50ng", cells=["Treg", "CD4+", "CD8+"])

    return f


def CoH_Bar_Plot(ax, tFac, CoH_Array, mode, numComps, plot_comps):
    """Plots bar plot for spec"""
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame()

    for i in range(0, numComps):
        tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, i], "Component": (i + 1), mode: mode_labels})])
    tFacDF = tFacDF.loc[tFacDF["Component"].isin(plot_comps)]
    sns.barplot(data=tFacDF, x=mode, y="Component_Val", hue="Component", ax=ax)
