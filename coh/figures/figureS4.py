"""
This creates Figure 8.
"""
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..flow import make_flow_df, make_CoH_Tensor, make_flow_sc_dataframe

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    CoH_sc_DF = make_flow_sc_dataframe()

    return f
