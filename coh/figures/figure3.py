"""
This creates Figure 3, dissection of signaling.
"""
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore, pearsonr
from .common import subplotLabel, getSetup, BC_scatter_cells, CoH_Scat_Plot
from ..flow import make_CoH_Tensor, get_status_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    ax[0].set(ylim=(-2, 8))
    ax[1].set(ylim=(-4, 4))
    ax[2].set(xlim=(-3, 2), ylim=(0, 5))
    ax[3].set(xlim=(0, 6), ylim=(0, 6))
    ax[4].set(xlim=(-3, 2), ylim=(-1, 4))
    ax[5].set(xlim=(-1, 4), ylim=(0, 6))
    ax[6].set(xlim=(0, 1), ylim=(0, 1))
    ax[7].set(xlim=(0, 1.5), ylim=(0, 1.5))
    ax[8].set(xlim=(-3, 2), ylim=(-2, 2))
    ax[9].set(xlim=(-3, 2), ylim=(-3, 1))
    ax[9].set(xlim=(0, 6), ylim=(0, 6))

    CoH_DF = pd.read_csv("./coh/data/CoH_Flow_DF.csv", index_col=0)
    treatments = np.array(["IL2-50ng", "IL4-50ng", "IL6-50ng", "IL10-50ng", "IFNg-50ng", "TGFB-50ng", "IFNg-50ng+IL6-50ng", "Untreated"])
    CoH_DF = CoH_DF.loc[(CoH_DF.Treatment.isin(treatments)) & (CoH_DF.Time == "15min") ].dropna()


    # Figure A Markers for signaling component
    cells = ["CD16 NK", "CD20 B", "CD4+", "CD8+", "CD33 Myeloid", "Treg"]
    DF = CoH_DF.loc[(CoH_DF.Marker == "pSTAT3") & (CoH_DF.Treatment != "Untreated")]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    DF = DF[(DF.Cell.isin(cells))]
    BC_scatter_cells(ax[0], DF, "pSTAT3", "IL10-50ng")

    # B Baseline pSTAT3

    DF = CoH_DF.loc[(CoH_DF.Marker == "pSTAT3") & (CoH_DF.Treatment == "Untreated")]
    DF["Mean"] -= np.mean(DF["Mean"].values)
    DF["Mean"] /= np.std(DF["Mean"].values)
    DF = DF[(DF.Cell.isin(cells))]
    BC_scatter_cells(ax[1], DF, "pSTAT3", "Untreated")

    # C Baseline pSTAT3 vs pSTAT3 induced

    meanDF = CoH_DF.groupby(["Patient", "Cell", "Treatment", "Marker"]).mean().reset_index()
    

    meanDF = (meanDF.pivot(index=["Patient", "Cell", "Treatment"], columns="Marker", values="Mean").reset_index().set_index("Patient"))
    treatedDF = meanDF.loc[meanDF.Treatment != "Untreated"]
    untreatedDF = meanDF.loc[meanDF.Treatment == "Untreated"]
    treatedDF.iloc[:, 2::] = treatedDF.iloc[:, 2::].apply(zscore)
    untreatedDF.iloc[:, 2::] = untreatedDF.iloc[:, 2::].apply(zscore)
    meanDF = pd.concat([treatedDF, untreatedDF])

    plot_by_patient(
        meanDF,
        cell1="CD8+",
        receptor1="pSTAT3",
        treatment1="Untreated",
        cell2="CD8+",
        receptor2="pSTAT3",
        treatment2="IL10-50ng",
        ax=ax[2],
    )
    print("HI4")

    # D CD8+ pSTAT3 vs B pSTAT3 in IL10

    plot_by_patient(
        meanDF,
        cell1="CD20 B",
        receptor1="pSTAT3",
        treatment1="IL10-50ng",
        cell2="CD8 TCM",
        receptor2="pSTAT3",
        treatment2="IL10-50ng",
        ax=ax[3],
    )
    print("HI5")
    
    # E CD8+ pSTAT3 vs B pSTAT3 in IL10

    plot_by_patient(
        meanDF,
        cell1="CD8+",
        receptor1="pSTAT5",
        treatment1="Untreated",
        cell2="CD8+",
        receptor2="pSTAT5",
        treatment2="IL2-50ng",
        ax=ax[4],
    )
    

    # F CD8+ pSTAT5 vs Treg pSTAT5 in IL2

    plot_by_patient(
        meanDF,
        cell1="CD8+",
        receptor1="pSTAT5",
        treatment1="IL2-50ng",
        cell2="Treg",
        receptor2="pSTAT5",
        treatment2="IL2-50ng",
        ax=ax[5],
    )

    # G CPD Components

    with open('./coh/data/signaling.pkl', 'rb') as ff:
        tFacAllM = pickle.load(ff) # 12 component
    CoH_Data = make_CoH_Tensor(just_signal=True)
    CoH_Scat_Plot(ax[6], tFacAllM, CoH_Data, "Patient", plot_comps=[5, 6], status_df=get_status_df())


    # H Untreated pSTAT4 and pSMAD1-2 by cell

    plot_diff_cell(meanDF, "pSmad1-2", "Untreated", "pSTAT4", "Untreated", ax[7])

    # I, J pSTAT4 vs pSMAD untreated per patient

    plot_by_patient(
        meanDF,
        cell1="CD8+",
        receptor1="pSTAT4",
        treatment1="Untreated",
        cell2="CD8+",
        receptor2="pSmad1-2",
        treatment2="Untreated",
        ax=ax[8],
    )

    plot_by_patient(
        meanDF,
        cell1="Treg",
        receptor1="pSTAT4",
        treatment1="Untreated",
        cell2="Treg",
        receptor2="pSmad1-2",
        treatment2="Untreated",
        ax=ax[9],
    )

    plot_by_patient(
        meanDF,
        cell1="Treg",
        receptor1="pSTAT5",
        treatment1="IL2-50ng",
        cell2="CD20 B",
        receptor2="pSTAT3",
        treatment2="IL10-50ng",
        ax=ax[10],
    )


    return f


def plot_by_patient(sigDF, cell1, receptor1, treatment1, cell2, receptor2, treatment2, ax):
    """Plots receptor in pop 1 vs receptor in pop 2 per patient, by disease status"""
    status_DF = get_status_df()
    plotDF = pd.DataFrame({"Patient": sigDF.loc[(sigDF.Cell == cell1) & (sigDF.Treatment == treatment1)].index.values})
    plotDF[cell1 + " " + receptor1 + " " + treatment1] = sigDF.loc[(sigDF.Cell == cell1) & (sigDF.Treatment == treatment1)][receptor1].values
    plotDF[cell2 + " " + receptor2 + " " + treatment2] = sigDF.loc[(sigDF.Cell == cell2) & (sigDF.Treatment == treatment2)][receptor2].values
    plotDF = plotDF.set_index("Patient").join(
        status_DF.set_index("Patient"), on="Patient"
    )
    sns.scatterplot(
        data=plotDF,
        x=cell1 + " " + receptor1 + " " + treatment1,
        y=cell2 + " " + receptor2 + " " + treatment2,
        ax=ax,
        hue="Status",
    )
    print(plotDF.corr())
    print(pearsonr(plotDF[cell1 + " " + receptor1 + " " + treatment1].values, plotDF[cell2 + " " + receptor2 + " " + treatment2].values))
    sns.regplot(data=plotDF, x=cell1 + " " + receptor1 + " " + treatment1, y=cell2 + " " + receptor2 + " " + treatment2, ax=ax, scatter=False, line_kws={"color": "gray"}, truncate=False)


def plot_diff_cell(sigDF, marker1, treatment1, marker2, treatment2, ax):
    """Plots receptor in pop 1 vs receptor in pop 2 per patient, by disease status"""
    status_DF = get_status_df()
    sigDF = sigDF.reset_index().set_index("Patient").join(status_DF.set_index("Patient"), on="Patient")
    plotDF = pd.DataFrame()
    for cell in sigDF.Cell.unique():
        BC_val_1 = np.mean(sigDF.loc[(sigDF.Status == "BC") & (sigDF.Cell == cell) & (sigDF.Treatment == treatment1)][marker1].values)
        BC_val_2 = np.mean(sigDF.loc[(sigDF.Status == "BC") & (sigDF.Cell == cell) & (sigDF.Treatment == treatment2)][marker2].values)
        Healthy_val_1 = np.mean(sigDF.loc[(sigDF.Status == "Healthy") & (sigDF.Cell == cell) & (sigDF.Treatment == treatment1)][marker1].values)
        Healthy_val_2 = np.mean(sigDF.loc[(sigDF.Status == "Healthy") & (sigDF.Cell == cell) & (sigDF.Treatment == treatment2)][marker2].values)
        plotDF = pd.concat([plotDF, pd.DataFrame({"Cell": [cell], "BC - Healthy Baseline " + marker1: BC_val_1 - Healthy_val_1, "BC - Healthy Baseline " + marker2: BC_val_2 - Healthy_val_2})])
    
    sns.scatterplot(data=plotDF, x="BC - Healthy Baseline " + marker1, y="BC - Healthy Baseline " + marker2, hue="Cell", style="Cell", ax=ax)
    print(plotDF.corr())
    print(pearsonr(plotDF["BC - Healthy Baseline " + marker1].values, plotDF["BC - Healthy Baseline " + marker2].values))
    sns.regplot(data=plotDF, x="BC - Healthy Baseline " + marker1, y="BC - Healthy Baseline " + marker2, ax=ax, scatter=False, line_kws={"color": "gray"}, truncate=False)