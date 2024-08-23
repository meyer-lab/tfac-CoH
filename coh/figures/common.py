"""This file contains functions that are used in multiple figures."""
from __future__ import annotations

import sys
import time
from string import ascii_lowercase
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn import preprocessing
from statannot import add_stat_annotation

from ..flow import get_status_dict
from ..flow_rec import get_status_dict_rec

if TYPE_CHECKING:
    import xarray as xa

mpl.use("AGG")

mpl.rcParams["legend.labelspacing"] = 0.2
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["xtick.major.pad"] = 1.0
mpl.rcParams["ytick.major.pad"] = 1.0
mpl.rcParams["xtick.minor.pad"] = 0.9
mpl.rcParams["ytick.minor.pad"] = 0.9
mpl.rcParams["legend.handletextpad"] = 0.5
mpl.rcParams["legend.handlelength"] = 0.5
mpl.rcParams["legend.framealpha"] = 0.5
mpl.rcParams["legend.markerscale"] = 0.7
mpl.rcParams["legend.borderpad"] = 0.35
mpl.rcParams["svg.fonttype"] = "none"


def getSetup(figsize: tuple[float, float], gridd: tuple[int, int], multz=None):
    """Establish figure set-up with subplots."""
    sns.set_theme(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    if multz is None:
        multz = {}

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = []
    while x < gridd[0] * gridd[1]:
        if x not in multz:  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz:  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def subplotLabel(axs) -> None:
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_lowercase[ii],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            fontsize=14,
        )


def genFigure() -> None:
    """Main figure generation function."""
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec("from coh.figures." + nameOut + " import makeFigure", globals())
    ff = makeFigure()  # noqa: F821
    ff.savefig(fdir + nameOut + ".svg", dpi=300, bbox_inches="tight", pad_inches=0)

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")



def plot_tFac_CoH(axs: list, tFac, CoH_Array: xa.DataArray) -> None:
    """Plots tensor factorization of cells."""
    for ii in range(CoH_Array.ndim):
        mode = CoH_Array.dims[ii]
        tFacDF = pd.DataFrame(
            tFac.factors[ii],
            index=CoH_Array.coords[mode],
            columns=[str(i + 1) for i in range(tFac.factors[ii].shape[1])],
        )

        cmap = sns.color_palette("vlag", as_cmap=True)
        sns.heatmap(data=tFacDF, ax=axs[ii], cmap=cmap, vmin=-1, vmax=1, cbar=(ii == 0))


def scatter_common(ax, hist_DF: pd.DataFrame, filter: bool) -> None:
    filt_cells = []
    pvals = []
    for cell in hist_DF.Cell.unique():
        BC_samps = hist_DF.loc[
            (hist_DF.Status == "BC") & (hist_DF.Cell == cell)
        ].Mean.values
        H_samps = hist_DF.loc[
            (hist_DF.Status == "Healthy") & (hist_DF.Cell == cell)
        ].Mean.values
        t_res = mannwhitneyu(BC_samps, H_samps)
        if t_res[1] < (0.05):
            filt_cells.append(cell)
            if t_res[1] < 0.0005:
                pvals.append("***")
            elif t_res[1] < 0.005:
                pvals.append("**")
            elif t_res[1] < 0.05:
                pvals.append("*")
            else:
                pvals.append("****")
        elif not filter:
            pass

    if filter:
        hist_DF = hist_DF.loc[hist_DF.Cell.isin(filt_cells)]

    boxpairs = []
    for cell in filt_cells:
        boxpairs.append([(cell, "Healthy"), (cell, "BC")])

    if len(filt_cells) > 0:
        add_stat_annotation(
            ax=ax,
            data=hist_DF,
            x="Cell",
            y="Mean",
            hue="Status",
            box_pairs=boxpairs,
            text_annot_custom=pvals,
            perform_stat_test=False,
            loc="inside",
            pvalues=np.tile(0, len(filt_cells)),
            verbose=0,
        )


def BC_scatter_cells(
    ax, CoH_DF: pd.DataFrame, marker: str, cytokine: str, filter=False,
) -> None:
    """Scatters specific responses."""
    CoH_DF = CoH_DF.loc[(CoH_DF.Time == "15min")]
    status_dict = get_status_dict()
    hist_DF = CoH_DF.loc[(CoH_DF.Treatment == cytokine) & (CoH_DF.Marker == marker)]
    hist_DF = hist_DF.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()
    hist_DF["Status"] = hist_DF.replace({"Patient": status_dict}).Patient.values

    sns.boxplot(data=hist_DF, y="Mean", x="Cell", hue="Status", showfliers=False, ax=ax)
    ax.set(title=marker + " in response to " + cytokine, ylabel=marker, xlabel="Status")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    scatter_common(ax, hist_DF, filter=filter)


def BC_scatter_cells_rec(ax, CoH_DF: pd.DataFrame, marker: str, filter=False) -> None:
    """Scatters specific responses."""
    status_dict = get_status_dict_rec()
    hist_DF = CoH_DF.loc[(CoH_DF.Marker == marker)]
    hist_DF = hist_DF.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()
    hist_DF["Status"] = hist_DF.replace({"Patient": status_dict}).Patient.values

    sns.boxplot(data=hist_DF, y="Mean", x="Cell", hue="Status", showfliers=False, ax=ax)
    ax.set(title=marker, ylabel=marker, xlabel="Status")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    scatter_common(ax, hist_DF, filter=filter)


def CoH_Scat_Plot(ax, tFac, CoH_Array, mode, plot_comps, status_df) -> None:
    """Plots bar plot for spec."""
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[1][coord]
    tFacDF = pd.DataFrame(
        mode_facs, index=mode_labels, columns=[i + 1 for i in range(mode_facs.shape[1])],
    )
    colors = sns.color_palette(n_colors=2)
    palette = {"BC": colors[0], "Healthy": colors[1]}

    if mode == "Patient":
        tFacDF = pd.concat([tFacDF, status_df.set_index("Patient")], axis=1)
        sns.scatterplot(
            data=tFacDF,
            x=plot_comps[0],
            y=plot_comps[1],
            hue="Status",
            palette=palette,
            ax=ax,
        )
    else:
        sns.scatterplot(data=tFacDF, x=plot_comps[0], y=plot_comps[1], ax=ax)
    ax.set(
        xlabel="Component " + str(plot_comps[0]),
        ylabel="Component " + str(plot_comps[1]),
    )


def comp_corr_plot(tFac, CoH_Array, status_DF, ax) -> None:
    """Plots correlation which each component has with outcome across patients."""
    coord = CoH_Array.dims.index("Patient")
    mode_facs = tFac[1][coord]
    Donor_CoH_y = preprocessing.label_binarize(
        status_DF.Status, classes=["Healthy", "BC"],
    ).flatten()
    corrDF = pd.DataFrame(data=mode_facs, columns=np.arange(1, mode_facs.shape[1] + 1))
    corrDF["BC Status"] = Donor_CoH_y
    corrDF = corrDF.corr()
    corrDF = corrDF.loc["BC Status", :].to_frame()
    corrDF = corrDF.drop("BC Status").reset_index()
    corrDF.columns = ["Component", "BC Correlation"]
    sns.barplot(data=corrDF, y="BC Correlation", x="Component", color="k", ax=ax)
    ax.set(ylim=(-1, 1), ylabel="Component", xlabel="Correlation with BC")
