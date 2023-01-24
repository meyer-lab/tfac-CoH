"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_lowercase
import sys
import time
import numpy as np
import seaborn as sns
import matplotlib
import svgutils.transform as st
import pandas as pd
from matplotlib import gridspec, pyplot as plt
from scipy.stats import ttest_ind
from statannot import add_stat_annotation
from ..tensor import get_status_dict, get_status_dict_rec

matplotlib.use("AGG")

matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35
matplotlib.rcParams["svg.fonttype"] = "none"


def getSetup(figsize, gridd, multz=None, empts=None):
    """Establish figure set-up with subplots."""
    sns.set(style="whitegrid", font_scale=0.7, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = {}

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = []
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x: x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def subplotLabel(axs):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(-0.2, 1.2, ascii_lowercase[ii], transform=ax.transAxes, fontweight="bold", va="top", fontsize=14)


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1):
    """Add cartoon to a figure file."""

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale=scalee)

    template.append(cartoon)
    template.save(figFile)


def genFigure():
    """Main figure generation function."""
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec("from coh.figures." + nameOut + " import makeFigure", globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=300, bbox_inches="tight", pad_inches=0)

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")


status_dict = {"Patient 26": "Healthy",
               "Patient 28": "Healthy",
               "Patient 30": "Healthy",
               "Patient 34": "Healthy",
               "Patient 35": "Healthy",
               "Patient 43": "Healthy",
               "Patient 44": "Healthy",
               "Patient 45": "Healthy",
               "Patient 52": "Healthy",
               "Patient 52A": "Healthy",
               "Patient 54": "Healthy",
               "Patient 56": "Healthy",
               "Patient 58": "Healthy",
               "Patient 60": "Healthy",
               "Patient 61": "Healthy",
               "Patient 62": "Healthy",
               "Patient 63": "Healthy",
               "Patient 66": "Healthy",
               "Patient 68": "Healthy",
               "Patient 69": "Healthy",
               "Patient 70": "Healthy",
               "Patient 79": "Healthy",
               "Patient 19186-4": "BC",
               "Patient 19186-8": "BC",
               "Patient 406": "BC",
               "Patient 19186-10-T1": "BC",
               "Patient 19186-10-T2": "BC",
               "Patient 19186-10-T3": "BC",
               "Patient 19186-15-T1": "BC",
               "Patient 19186-15-T2": "BC",
               "Patient 19186-15-T3": "BC",
               "Patient 19186-2": "BC",
               "Patient 19186-3": "BC",
               "Patient 19186-14": "BC",
               "Patient 21368-3": "BC",
               "Patient 21368-4": "BC"}


def make_status_DF():
    statusDF = pd.DataFrame.from_dict(status_dict, orient='index').reset_index()
    statusDF.columns = ["Patient", "Status"]
    statusDF.to_csv("coh/data/Patient_Status.csv")


def BC_scatter(ax, CoH_DF, marker, cytokine, cells=False):
    """Scatters specific responses"""
    CoH_DF = CoH_DF.loc[(CoH_DF.Time == "15min")]
    if not cells:
        hist_DF = CoH_DF.loc[(CoH_DF.Treatment == cytokine) & (CoH_DF.Marker == marker)]
    else:
        hist_DF = CoH_DF.loc[(CoH_DF.Treatment == cytokine) & (CoH_DF.Marker == marker) & (CoH_DF["Cell"].isin(cells))]
    hist_DF = hist_DF.groupby(["Patient", "Marker"]).Mean.mean().reset_index()
    hist_DF["Status"] = hist_DF.replace({"Patient": status_dict}).Patient.values

    sns.boxplot(data=hist_DF, y="Mean", x="Status", ax=ax)
    ax.set(title=marker + " in response to " + cytokine, ylabel=marker, xlabel="Status")
    add_stat_annotation(ax=ax, data=hist_DF, x="Status", y="Mean", test='t-test_ind', order=["Healthy", "BC"], box_pairs=[("Healthy", "BC")], text_format='full', loc='inside', verbose=2)


def BC_scatter_cells(ax, CoH_DF, marker, cytokine, filter=False):
    """Scatters specific responses"""
    CoH_DF = CoH_DF.loc[(CoH_DF.Time == "15min")]
    status_dict = get_status_dict()
    hist_DF = CoH_DF.loc[(CoH_DF.Treatment == cytokine) & (CoH_DF.Marker == marker)]
    hist_DF = hist_DF.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()
    hist_DF["Status"] = hist_DF.replace({"Patient": status_dict}).Patient.values


    filt_cells = []
    pvals = []
    for cell in hist_DF.Cell.unique():
        BC_samps = hist_DF.loc[(hist_DF.Status == "BC") & (hist_DF.Cell == cell)].Mean.values
        H_samps = hist_DF.loc[(hist_DF.Status == "Healthy") & (hist_DF.Cell == cell)].Mean.values
        t_res = ttest_ind(BC_samps, H_samps)
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
        else:
            if not filter:
                pvals.append("ns")
    if filter:
        hist_DF = hist_DF.loc[hist_DF.Cell.isin(filt_cells)]

    sns.boxplot(data=hist_DF, y="Mean", x="Cell", hue="Status", ax=ax)
    ax.set(title=marker + " in response to " + cytokine, ylabel=marker, xlabel="Status")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    boxpairs = []
    for cell in hist_DF.Cell.unique():
        boxpairs.append([(cell, "Healthy"), (cell, "BC")])
    if filter:
        add_stat_annotation(ax=ax, data=hist_DF, x="Cell", y="Mean", hue="Status", box_pairs=boxpairs, text_annot_custom=pvals, perform_stat_test=False, loc='inside', pvalues=np.tile(0, len(filt_cells)), verbose=0)
    else:
        add_stat_annotation(ax=ax, data=hist_DF, x="Cell", y="Mean", hue="Status", box_pairs=boxpairs, text_annot_custom=pvals, perform_stat_test=False, loc='inside', pvalues=np.tile(0, len(hist_DF.Cell.unique())), verbose=0)
    # ad
    # add_stat_annotation(ax=ax, data=hist_DF, x="Cell", y="Mean", hue="Status", test='t-test_ind', box_pairs=boxpairs, text_format='star', loc='inside', verbose=2)


def BC_scatter_cells_rec(ax, CoH_DF, marker, filter=False):
    """Scatters specific responses"""
    status_dict = get_status_dict_rec()
    hist_DF = CoH_DF.loc[(CoH_DF.Marker == marker)]
    hist_DF = hist_DF.groupby(["Cell", "Patient", "Marker"]).Mean.mean().reset_index()
    hist_DF["Status"] = hist_DF.replace({"Patient": status_dict}).Patient.values


    filt_cells = []
    pvals = []
    for cell in hist_DF.Cell.unique():
        BC_samps = hist_DF.loc[(hist_DF.Status == "BC") & (hist_DF.Cell == cell)].Mean.values
        H_samps = hist_DF.loc[(hist_DF.Status == "Healthy") & (hist_DF.Cell == cell)].Mean.values
        t_res = ttest_ind(BC_samps, H_samps)
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
        else:
            if not filter:
                pvals.append("ns")
    if filter:
        hist_DF = hist_DF.loc[hist_DF.Cell.isin(filt_cells)]
    sns.boxplot(data=hist_DF, y="Mean", x="Cell", hue="Status", ax=ax)
    ax.set(title=marker, ylabel=marker, xlabel="Status")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    boxpairs = []
    for cell in hist_DF.Cell.unique():
        boxpairs.append([(cell, "Healthy"), (cell, "BC")])
    if filter:
        add_stat_annotation(ax=ax, data=hist_DF, x="Cell", y="Mean", hue="Status", box_pairs=boxpairs, text_annot_custom=pvals, perform_stat_test=False, loc='inside', pvalues=np.tile(0, len(filt_cells)), verbose=0)
    else:
        add_stat_annotation(ax=ax, data=hist_DF, x="Cell", y="Mean", hue="Status", box_pairs=boxpairs, text_annot_custom=pvals, perform_stat_test=False, loc='inside', pvalues=np.tile(0, len(hist_DF.Cell.unique())), verbose=0)
