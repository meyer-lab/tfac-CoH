"""This creates Figure 5, dissection of receptor data."""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import zscore
from sklearn import metrics, preprocessing

from ..flow_rec import get_status_rec_df, make_CoH_Tensor_rec
from ..tensor import factorTensor, lrmodel
from .common import getSetup, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((9, 6.75), (3, 4))

    # Add subplot labels
    subplotLabel(ax)
    ax[0].set(xlim=(-1, 3), ylim=(-6, 2))
    ax[1].set(xlim=(-1, 3), ylim=(-3, 2))
    ax[2].set(ylim=(-3, 2))
    ax[3].set(ylim=(0, 4))
    ax[4].set(xlim=(-2, 0.5), ylim=(-3, 1))
    ax[5].set(xlim=(-2.5, 0), ylim=(-3, 1))
    ax[6].set(xlim=(1, 3), ylim=(-2, 1))
    ax[7].set(xlim=(1, 3), ylim=(-3, 1))

    CoH_DF = pd.read_csv("./coh/data/CoH_Rec_DF.csv", index_col=0)

    # Make mean Z scored DF
    meanDF = CoH_DF.groupby(["Patient", "Cell", "Marker"]).mean().reset_index()

    meanDF = (
        meanDF.pivot(index=["Patient", "Cell"], columns="Marker", values="Mean")
        .reset_index()
        .set_index("Patient")
    )
    meanDF.loc[:, meanDF.columns.values != "Cell"] = meanDF.loc[
        :, meanDF.columns.values != "Cell",
    ].apply(zscore)

    # E PD-L1 in B vs CD8 Cells

    plot_by_patient(
        meanDF,
        cell1="CD8 Naive",
        receptor1="IL2RB",
        cell2="CD8 Naive",
        receptor2="IL12RI",
        ax=ax[0],
    )

    plot_by_patient(
        meanDF,
        cell1="CD8 Naive",
        receptor1="IL2RB",
        cell2="CD20 B",
        receptor2="IL12RI",
        ax=ax[1],
    )

    plot_by_patient(
        meanDF,
        cell1="CD8 TEM",
        receptor1="PD_L1",
        cell2="CD20 B",
        receptor2="PD_L1",
        ax=ax[4],
    )

    # F IL6Ra in B vs CD8 Cells

    plot_by_patient(
        meanDF,
        cell1="CD8 TEM",
        receptor1="IL6Ra",
        cell2="CD20 B",
        receptor2="IL6Ra",
        ax=ax[5],
    )

    # G IL2Ra Tregs vs PD-L1 CD8s

    plot_by_patient(
        meanDF,
        cell1="Treg",
        receptor1="IL2Ra",
        cell2="CD8 TEM",
        receptor2="PD_L1",
        ax=ax[6],
    )

    # H IL2Ra Tregs vs IL6Ra B

    plot_by_patient(
        meanDF,
        cell1="Treg",
        receptor1="IL2Ra",
        cell2="CD20 B",
        receptor2="IL6Ra",
        ax=ax[7],
    )

    # I Univariate vs coordinated ROC

    CoH_Data = make_CoH_Tensor_rec()
    tFacAllM = factorTensor(CoH_Data.to_numpy(), r=5)
    mode = CoH_Data.dims[0]
    tFacDF = pd.DataFrame(
        tFacAllM.factors[0],
        index=CoH_Data.coords[mode],
        columns=[str(i + 1) for i in range(tFacAllM.factors[0].shape[1])],
    )

    AUC_DF = ROC_plot(
        meanDF,
        receptors=["IL6Ra", "IL6Ra", "PD_L1", "PD_L1", "IL2Ra"],
        cells=["CD8 TEM", "CD20 B", "CD8 TEM", "CD20 B", "Treg"],
        tFacDF=tFacDF,
        comp=2,
        ax=ax[8],
    )

    plot_AUC_bar(AUC_DF, ax[9])

    # ax[8].set(xlim=(0, 1), ylim=(0, 1))

    return f


def plot_by_patient(recDF, cell1, receptor1, cell2, receptor2, ax) -> None:
    """Plots receptor in pop 1 vs receptor in pop 2 per patient, by disease status."""
    status_DF = get_status_rec_df()
    plotDF = pd.DataFrame({"Patient": recDF.loc[recDF.Cell == cell1].index.values})
    plotDF[cell1 + " " + receptor1] = recDF.loc[recDF.Cell == cell1][receptor1].values
    plotDF[cell2 + " " + receptor2] = recDF.loc[recDF.Cell == cell2][receptor2].values
    plotDF = plotDF.set_index("Patient").join(
        status_DF.set_index("Patient"), on="Patient",
    )
    sns.scatterplot(
        data=plotDF,
        x=cell1 + " " + receptor1,
        y=cell2 + " " + receptor2,
        ax=ax,
        hue="Status",
    )
    sns.regplot(
        data=plotDF,
        x=cell1 + " " + receptor1,
        y=cell2 + " " + receptor2,
        ax=ax,
        scatter=False,
        line_kws={"color": "gray"},
        truncate=False,
    )


def ROC_plot(recDF, receptors, cells, tFacDF, comp, ax):
    """Plots accuracy of classification using receptors and a tfac component."""
    status_DF = get_status_rec_df()
    AUC_DF = pd.DataFrame()

    for i, receptor in enumerate(receptors):
        predDF = recDF.loc[recDF.Cell == cells[i]].reset_index()[["Patient", receptor]]
        predDF = predDF.set_index("Patient").join(
            status_DF.set_index("Patient"), on="Patient",
        )
        Donor_CoH_y = preprocessing.label_binarize(
            predDF.Status.values, classes=["Healthy", "BC"],
        ).flatten()
        LR_CoH = lrmodel.fit(stats.zscore(predDF[receptor][:, np.newaxis]), Donor_CoH_y)
        y_pred = LR_CoH.predict_proba(stats.zscore(predDF[receptor][:, np.newaxis]))[
            :, 1,
        ]
        fpr, tpr, _ = metrics.roc_curve(Donor_CoH_y, y_pred)
        auc = round(metrics.roc_auc_score(Donor_CoH_y, y_pred), 4)
        ax.plot(fpr, tpr, label=cells[i] + " " + receptor)
        AUC_DF = pd.concat(
            [AUC_DF, pd.DataFrame({"Feature": cells[i] + " " + receptor, "AUC": [auc]})],
        )

    predDF = tFacDF[str(comp)].reset_index()
    predDF.columns = ["Patient", "Comp. " + str(comp)]
    predDF = predDF.set_index("Patient").join(
        status_DF.set_index("Patient"), on="Patient",
    )
    Donor_CoH_y = preprocessing.label_binarize(
        predDF.Status.values, classes=["Healthy", "BC"],
    ).flatten()
    LR_CoH = lrmodel.fit(
        stats.zscore(predDF["Comp. " + str(comp)][:, np.newaxis]), Donor_CoH_y,
    )
    y_pred = LR_CoH.predict_proba(
        stats.zscore(predDF["Comp. " + str(comp)][:, np.newaxis]),
    )[:, 1]
    fpr, tpr, _ = metrics.roc_curve(Donor_CoH_y, y_pred)
    auc = round(metrics.roc_auc_score(Donor_CoH_y, y_pred), 4)
    ax.plot(fpr, tpr, label="Comp. " + str(comp) + ", AUC=" + str(auc))
    AUC_DF = pd.concat(
        [AUC_DF, pd.DataFrame({"Feature": "Comp. " + str(comp), "AUC": [auc]})],
    )

    ax.legend()
    return AUC_DF


def plot_AUC_bar(AUC_DF, ax) -> None:
    """Plots AUC from AUC analysis."""
    sns.barplot(data=AUC_DF, x="Feature", y="AUC", ax=ax)
    ax.set(ylim=(0.5, 1))
