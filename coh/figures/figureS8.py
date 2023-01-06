"""
This creates Figure S8, full panel of receptor comps.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statannot import add_stat_annotation
from scipy.stats import ttest_ind
from tensorly.cp_tensor import cp_flip_sign
from tensorpack.cmtf import cp_normalize, perform_CP
from .figureCommon import subplotLabel, getSetup, BC_scatter_cells_rec
from os.path import join, dirname
from ..flow_rec import make_flow_df_rec, make_CoH_Tensor_rec
from ..tensor import CoH_LogReg_plot, BC_status_plot_rec, get_status_dict_rec, make_allrec_DF

plt.rcParams['svg.fonttype'] = 'none'
path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 9), (4, 3))

    # Add subplot labels
    subplotLabel(ax)
    # make_flow_df_rec()
    # make_CoH_Tensor_rec()
    CoH_Data_DF = pd.read_csv(join(path_here, "data/CoH_Rec_DF.csv"))
    for i, rec in enumerate(np.array(["IFNg R1", "TGFB RII", "PD1", "PD_L1", "IL2Ra", "IL2RB", "IL4Ra", "IL6Ra", "IL6RB", "IL7Ra", "IL10R", "IL12RI"])):
        DF = CoH_Data_DF.loc[CoH_Data_DF.Marker == rec]
        DF["Mean"] -= np.mean(DF["Mean"].values)
        DF["Mean"] /= np.std(DF["Mean"].values)
        BC_scatter_cells_rec(ax[i], DF, rec, filter=False)

    return f
