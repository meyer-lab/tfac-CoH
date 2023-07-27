"""
This creates Figure 8.
"""
from .common import subplotLabel, getSetup
from os.path import dirname
from ..flow import make_flow_sc_dataframe

path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (3, 2))

    # Add subplot labels
    subplotLabel(ax)

    CoH_sc_DF = make_flow_sc_dataframe()

    return f
