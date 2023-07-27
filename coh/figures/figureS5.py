"""
This creates Figure S5, scattering receptor data.
"""
import numpy as np
import pandas as pd
from .common import subplotLabel, getSetup, BC_scatter_cells_rec
from os.path import join, dirname

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), (2, 3))

    # Add subplot labels
    subplotLabel(ax)
    CoH_Data_DF_R = pd.read_csv(join(path_here, "data/CoH_Rec_DF.csv"))
    filt_list = [False, True, True, True, True, True]

    for i, rec in enumerate(np.array(["IL10R", "IL2RB", "IL12RI", "TGFB RII", "PD_L1", "IL6Ra"])):
        BC_scatter_cells_rec(ax[i], CoH_Data_DF_R, rec, filter=filt_list[i])

    return f
