from tensorly.regression import cp_plsr
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score
from importlib.abc import PathEntryFinder
import os
from os.path import dirname, join
from types import CellType
import pandas as pd
import numpy as np
import warnings
import xarray as xa
from copy import copy
import tensorly
import matplotlib.pyplot as plt
import seaborn as sns


path_here = os.path.dirname(os.path.dirname("__file__"))
warnings.filterwarnings("ignore")

## Pre-load the data for analysis

CoH_DF_PLS=xa.open_dataarray("/data/CoH Tensor DataSet for PLS.nc")
cancer_treatment = np.array(['Untreated', 'IFNg-50ng',  'IL10-50ng','IL2-50ng', 'IL4-50ng', 'IL6-50ng'])



total_treatment = np.array(CoH_DF_PLS.Treatment)

cancer_treatment_indicies=np.argwhere(np.isin(total_treatment,cancer_treatment)).ravel()

cancer_treatment_indicies=np.array(cancer_treatment_indicies)

CoH_DF_PLS=CoH_DF_PLS[:,0,cancer_treatment_indicies,:,:]

patient_response_ori = get_status_df()



def find_best_number_of_components_for_PLS(CoH_DF_PLS,patient_response_original):

    ## Find the most optimal number of componenets for tPLS decomposition. Logistic regression was included on top of PLSR because PLSR is 
    ## doing regression to approximate the binary label (with cancer as 1 and without cancer/healthy as 0) rather than classification.

    endpoint_cancer_or_not_for_each_pt = patient_response_original
    patient_response = pd.get_dummies(endpoint_cancer_or_not_for_each_pt,columns=["Status"])
    patient_response = patient_response[["Patient","Status_BC"]] #Healthy:0, BC:1
    patient_response_DF =xa.DataArray(patient_response["Status_BC"].astype("float64"),dims=("Patient"),coords={"Patient": patient_response["Patient"]})

    skf=StratifiedKFold(n_splits=10,shuffle=True,random_state=0)

    X=patient_response["Patient"]
    y=patient_response["Status_BC"]

    k_component_performance_accuracy=[]
    k_component_performance_roc=[]

    for k in range(1,21):
        print("Let's test with ", k, " components for our exciting PLS")
        cv_10_fold_accuracy = []
        cv_10_fold_roc = []

        for i, (train_index,test_index) in enumerate(skf.split(X,y)):

            print("Currently it is ", i+1, " fold.")

            CoH_PLSR = cp_plsr.CP_PLSR(n_components=k,tol=1e-7,n_iter_max=4000)
            CoH_PLSR.fit(CoH_DF_PLS[train_index,:,:,:],patient_response_DF[train_index])
            print("Fit done")
            CoH_PLSR_train_transformed=CoH_PLSR.transform(CoH_DF_PLS[train_index,:,:,:],patient_response_DF[train_index])[0]
            CoH_PLSR_test_transformed=CoH_PLSR.transform(CoH_DF_PLS[test_index,:,:,:],patient_response_DF[test_index])[0]

            logistic = LR()
            logistic.fit(CoH_PLSR_train_transformed,patient_response_DF[train_index])

            accuracy = logistic.score(CoH_PLSR_test_transformed,patient_response_DF[test_index])
            CoH_PLSR_test_decision_function = logistic.decision_function(CoH_PLSR_test_transformed)
            roc_auc = roc_auc_score(patient_response_DF[test_index],CoH_PLSR_test_decision_function,average="weighted")

            cv_10_fold_accuracy.append(accuracy)
            cv_10_fold_roc.append(roc_auc)

            print(k, " Compoenet, ", i+1, " Fold ACCURACY: ",accuracy)
            print(k, " Compoenet, ", i+1, " Fold ROC: ",roc_auc)

        print(k, " Compoenet, ", " Mean ACCURACY: ",np.mean(cv_10_fold_accuracy))
        print(k, " Compoenet, ", " Mean ROC: ",np.mean(cv_10_fold_roc))
        k_component_performance_accuracy.append(np.mean(cv_10_fold_accuracy))
        k_component_performance_roc.append(np.mean(cv_10_fold_roc))

    k_component_performance_accuracy=np.array(k_component_performance_accuracy)
    k_component_performance_roc=np.array(k_component_performance_roc)

    performance_result=pd.DataFrame()
    performance_result["Component"] = np.arange(1,21)
    performance_result["Accuracy"] = k_component_performance_accuracy
    performance_result["ROC_AUC"] = k_component_performance_roc

    performance_result.to_csv("k_compoenent_PLS_result_LR_15min_trial.csv")


    best_k_accuracy = performance_result["Component"][np.argmax(performance_result["Accuracy"])]
    best_k_roc = performance_result["Component"][np.argmax(performance_result["ROC_AUC"])]

    print("The ", best_k_accuracy, " components achieves the best accuracy of ", np.max(performance_result["Accuracy"]))
    print("The ", best_k_roc, " components achieves the best roc_auc of ", np.max(performance_result["ROC_AUC"]))
    fig,ax=plt.subplots(1,2,figsize=(18,8))

    ax[0].set(xlabel = "Components",ylabel = "Mean Accuracy",xticks=np.arange(0,21,1))

    ax[0].plot(performance_result["Component"],performance_result["Accuracy"])

    ax[1].set(xlabel = "Components",ylabel = "Mean ROC-AUC",xticks=np.arange(0,21,1))

    ax[1].plot(performance_result["Component"],performance_result["ROC_AUC"])

    plt.tight_layout()

    fig.show()

    fig.savefig("Performance_Plot.png")


def weight_of_the_ith_component_for_PLS(CoH_DF_PLS,patient_response_original,numComps):

    ##Plot the figure for the weight of each component (in Logistic Regression) after tPLS with numComps' components 

    endpoint_cancer_or_not_for_each_pt = patient_response_original
    patient_response = pd.get_dummies(endpoint_cancer_or_not_for_each_pt,columns=["Status"])
    patient_response = patient_response[["Patient","Status_BC"]] #Healthy:0, BC:1
    patient_response_DF =xa.DataArray(patient_response["Status_BC"].astype("float64"),dims=("Patient"),coords={"Patient": patient_response["Patient"]})

    CoH_PLSR = cp_plsr.CP_PLSR(n_components=numComps,tol=1e-7,n_iter_max=4000)
    CoH_PLSR.fit(CoH_DF_PLS[:,:,:,:],patient_response_DF)
    print("Fit done")
    CoH_PLSR_all_transformed=CoH_PLSR.transform(CoH_DF_PLS[:,:,:,:],patient_response_DF[:])[0]

    logistic = LR()
    logistic.fit(CoH_PLSR_all_transformed,patient_response_DF[:])

    fig,ax=plt.subplots(1)
    coeff=logistic.coef_[0]
    ax.set(xlabel="The ith Component",ylabel="Weight",xticks=np.arange(1,numComps+1,1))
    ax.bar(np.arange(1,numComps+1,1),coeff)
    fig.savefig("Weight_of_the_ith_component.png")


def plot_heatmap_based_on_mode_by_PLS(CoH_DF_PLS,patient_response_original,mode,numComps):

    ## Adapted from Brian's figure3hm.py code. Plotting the heat map based on tPLS decomposition with numComps' Componenets. 

    ## The CP_PLSR() in the tensorly package had to be edited so that the X.factors is a class variable of CP_PLSR().

    endpoint_cancer_or_not_for_each_pt = patient_response_original
    patient_response = pd.get_dummies(endpoint_cancer_or_not_for_each_pt,columns=["Status"])
    patient_response = patient_response[["Patient","Status_BC"]] #Healthy:0, BC:1
    patient_response_DF =xa.DataArray(patient_response["Status_BC"].astype("float64"),dims=("Patient"),coords={"Patient": patient_response["Patient"]})

    CoH_PLSR = cp_plsr.CP_PLSR(n_components=numComps,tol=1e-7,n_iter_max=4000)
    CoH_PLSR.fit(CoH_DF_PLS[:,:,:,:],patient_response_DF)

    tFac=np.array(CoH_PLSR.X_factors)
    CoH_Array = CoH_DF_PLS
    mode_labels = CoH_Array[mode]
    coord = CoH_Array.dims.index(mode)
    mode_facs = tFac[coord]
    numComps=3
    tFacDF = pd.DataFrame()
    for i in range(0, numComps):
        tFacDF = pd.concat([tFacDF, pd.DataFrame({"Component_Val": mode_facs[:, i], "Component": (i + 1), mode: mode_labels})])

    tFacDF = pd.pivot(tFacDF, index="Component", columns=mode, values="Component_Val")
    cmap = sns.color_palette("vlag", as_cmap=True)
    status_df = patient_response_original.sort_values(by="Patient").reset_index()
    status = status_df["Status"]
    lut = dict(zip(status.unique(), "rbg"))
    col_colors = pd.DataFrame(status.map(lut))
    col_colors["Patient"] = status_df.Patient.values
    col_colors = col_colors.set_index("Patient")
    f = sns.clustermap(data=tFacDF, robust=True, cmap=cmap, vmin=-1, vmax=1, row_cluster=False, col_colors=col_colors, figsize=(19, 3))
    
    file_title=mode+" Heatmap from PLS.png"

    f.figure.savefig(file_title)



