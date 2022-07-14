"""
This creates Figure 1.
"""
import xarray as xa
import tensorly as tl
import numpy as np
import seaborn as sns
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
from tensorpack import Decomposition
from tensorpack.tucker import tucker_decomp
from tensorpack.plot import reduction, tucker_reduction
from .figureCommon import subplotLabel, getSetup
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from os.path import join

path_here = os.path.dirname(os.path.dirname(__file__))


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 3), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    CoH_Data = xa.open_dataarray(join(path_here, "data/CoHTensorDataJustSignal.nc"))
    
    #makePCA_df(CoH_Data)
    #plot_PCA(ax[0:2])
    
    # perform parafac
    tc = Decomposition(CoH_Data.to_numpy()[0:-9, :, :, :, :], max_rr=12)
    tc.perform_tfac()
    tc.perform_PCA(flattenon=2)

    reduction(ax[0], tc)
    tuck = Decomposition(CoH_Data.to_numpy()[0:-9, :, :, :, :], method=tucker_decomp)
    para = Decomposition(CoH_Data.to_numpy()[0:-9, :, :, :, :])
    tucker_reduction(ax[1], tuck, para)

    return f


def makePCA_df(TensorArray):
    """Returns PCA with score and loadings of COH DataSet"""
    DF = TensorArray.to_dataframe(name="value").reset_index()
    DF = DF.loc[(DF.Patient != "Patient 4") & (DF.Patient != "Patient 8") & (DF.Patient != "Patient 406")]
    PCAdf = pd.DataFrame()
    for patient in DF.Patient.unique():
        patientDF = DF.loc[DF.Patient == patient]
        patientRow = pd.DataFrame({"Patient": [patient]})
        for time in DF.Time.unique():
            for treatment in DF.Treatment.unique():
                for marker in DF.Marker.unique():
                    for cell in DF.Cell.unique():
                        uniqueDF = patientDF.loc[(patientDF.Time == time) & (patientDF.Marker == marker) & (patientDF.Treatment == treatment) & (patientDF.Cell == cell)]
                        patientRow[time + "_" + treatment + "_"+ marker + "_"+ cell] = uniqueDF.value.values
        PCAdf = pd.concat([PCAdf, patientRow])
    PCAdf.to_csv(join(path_here, "data/CoH_PCA.csv"))


def plot_PCA(ax):
    """Plots CoH PCA"""
    DF = pd.read_csv(join(path_here, "data/CoH_PCA.csv")).set_index("Patient").drop("Unnamed: 0", axis=1)
    pcaMat = DF.to_numpy()
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    pcaMat = scaler.fit_transform(pcaMat)
    scores = pca.fit_transform(pcaMat)
    loadings = pca.components_

    
    scoresDF = pd.DataFrame({"Patient": DF.index.values, "Component 1": scores[:, 0], "Component 2": scores[:, 1]})
    loadingsDF = pd.DataFrame()
    for i, col in enumerate(DF.columns):
        vars = col.split("_")
        loadingsDF = pd.concat([loadingsDF, pd.DataFrame({"Time": [vars[0]], "Treatment": vars[1], "Marker": vars[2], "Cell": vars[3], "Component 1": loadings[0, i], "Component 2": loadings[1, i]})])
    
    sns.scatterplot(data=scoresDF, hue="Patient", x="Component 1", y="Component 2", ax=ax[0])
    sns.scatterplot(data=loadingsDF, x="Component 1", y="Component 2", hue="Treatment", style="Cell", size="Marker", ax=ax[1])
