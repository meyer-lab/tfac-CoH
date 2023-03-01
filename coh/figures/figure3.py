"""
This creates Figure 3, classification analysis.
"""
import xarray as xa
import seaborn as sns
import pandas as pd
from .figureCommon import subplotLabel, getSetup
from os.path import join, dirname
from ..tensor import factorTensor, CoH_LogReg_plot, plot_tFac_CoH, make_alldata_DF, BC_status_plot


path_here = dirname(dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 6), (3, 4), multz={0: 3})

    # Add subplot labels
    subplotLabel(ax)
    #make_flow_df()
    # make_CoH_Tensor(basal=True)
    ax[0].axis("off")

    num_comps = 4
    CoH_Data = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc"))
    tFacAllM, _ = factorTensor(CoH_Data.values, numComps=num_comps)

    # make_alldata_DF(CoH_Data, PCA=False, basal=False)

    # matrix_DF = pd.read_csv(join(path_here, "data/CoH_Matrix.csv"), index_col=0).dropna(axis='columns').set_index("Patient")
    BC_status_plot(8, CoH_Data, ax[1])
    CoH_LogReg_plot(ax[2], tFacAllM, CoH_Data, num_comps)
    CoH_Scat_Plot(ax[3], tFacAllM, CoH_Data, "Patient", numComps=num_comps, plot_comps=[3, 4])
    plot_tFac_CoH(ax[4], tFacAllM, CoH_Data, "Marker", numComps=num_comps, cbar=False)

    num_comps = 4
    CoH_Data_B = xa.open_dataarray(join(path_here, "data/CoH_Tensor_DataSet_Basal.nc"))
    tFacAllM_B, _ = factorTensor(CoH_Data_B.values, numComps=num_comps)

    # make_alldata_DF(CoH_Data_B, PCA=False, basal=True)

    # matrix_DF_B = pd.read_csv(join(path_here, "data/CoH_Matrix_Basal.csv"), index_col=0).dropna(axis='columns').set_index("Patient")
    BC_status_plot(5, CoH_Data_B, ax[5], basal=True)
    CoH_LogReg_plot(ax[6], tFacAllM_B, CoH_Data_B, num_comps)
    CoH_Scat_Plot(ax[7], tFacAllM_B, CoH_Data_B, "Patient", numComps=num_comps, plot_comps=[1, 2])
    plot_tFac_CoH(ax[8], tFacAllM_B, CoH_Data_B, "Marker", numComps=num_comps, cbar=False)

    return f


def CoH_Scat_Plot(ax, tFac, CoH_Array, mode, numComps, plot_comps):
    """Plots bar plot for spec"""
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame()

    for i in range(0, numComps):
        tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, i], "Component": (i + 1), mode: mode_labels})])
    tFacDF = tFacDF.loc[tFacDF["Component"].isin(plot_comps)]
    tFacDF = tFacDF.pivot(index=mode, columns='Component', values='Component_Val')
    if mode == "Patient":
        status_df = pd.read_csv(join(path_here, "data/Patient_Status.csv")).set_index("Patient")
        tFacDF = pd.concat([tFacDF, status_df], axis=1)
        sns.scatterplot(data=tFacDF, x=plot_comps[0], y=plot_comps[1], hue="Status", style="Status", ax=ax)
    else:
        sns.scatterplot(data=tFacDF, x=plot_comps[0], y=plot_comps[1], ax=ax)
    ax.set(xlabel="Component " + str(plot_comps[0]), ylabel="Component " + str(plot_comps[1]))
