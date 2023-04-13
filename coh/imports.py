"""File that deals with everything about importing and sampling."""
import os
from functools import lru_cache
from os.path import join
import numpy as np
import pandas as pd
import numpy as np
import anndata as ad
from scipy.io import mmread
from scipy.sparse import coo_matrix
from zipfile import ZipFile



path_here = os.path.dirname(os.path.dirname(__file__))


def importCITE():
    """Downloads all surface markers and cell types"""
    CITEmarkerDF = pd.read_csv(join(path_here, "/opt/CoH/SingleCell/CITEdata_SurfMarkers.zip"))
    return CITEmarkerDF


def makeRNAseqDF(surface=False):
    """Makes surface RNAseq DF"""
    matrix = mmread(join(path_here, "coh/data/SC_seq/GSM5008737_RNA_3P-matrix.mtx.gz"))
    surfGenes = pd.read_csv("coh/data/SC_seq/SurfaceGenes.csv")
    featuresGenes = pd.read_csv("coh/data/SC_seq/RNAfeatures.csv")
    surfaceList = surfGenes["Gene"].values
    allFeatures = featuresGenes["Gene"].values

    if surface:
        featInd = np.isin(allFeatures, surfaceList)
    else: 
        featInd = np.isin(allFeatures, allFeatures)

    cols = [i for i, x in enumerate(featInd) if x]
    dataCoords = np.isin(matrix.row, cols)
    locList = np.where(featInd == True)[0].tolist()
    newCoords = np.arange(0, len(locList)).tolist()

    colDict = {}
    for key in locList:
        for value in newCoords:
            colDict[key] = value
            newCoords.remove(value)
            break

    def vec_translate(a, my_dict):
        return np.vectorize(my_dict.__getitem__)(a)

    matrix2 = coo_matrix((matrix.data[dataCoords], (matrix.col[dataCoords], vec_translate(matrix.row[dataCoords], colDict))), shape=(matrix.shape[1], np.count_nonzero(featInd)))
    geneCols = allFeatures[featInd]
    GeneDF = pd.DataFrame(data=matrix2.toarray(), columns=geneCols)
    cellTypesDF = pd.read_csv(join(path_here, "/opt/CoH/SingleCell/CITEcellTypes.csv"))
    GeneDF = pd.concat([GeneDF, cellTypesDF], axis=1)
    if surface:
        GeneDF.to_csv(join(path_here, "/opt/CoH/SingleCell/RNAseqSurface.csv.zip"))
    else:
        GeneDF.to_csv(join(path_here, "/opt/CoH/SingleCell/RNAseq.csv.zip"))


def makeRNAseqDF_Ann(surface=False):
    """Read data and make AnnData object"""
    RNA = ad.read_h5ad("/opt/CoH/SingleCell/CITE_RNA_raw.h5ad").transpose()
    cellTypesDF = pd.read_csv(join(path_here, "coh/data/SC_seq/CITEcellTypes.csv"))
    featuresGenes = pd.read_csv("coh/data/SC_seq/RNAfeatures.csv")
    surfGenes = pd.read_csv("coh/data/SC_seq/SurfaceGenes.csv")

    RNA.obs["CellType1"] = cellTypesDF.CellType1.values
    RNA.obs["CellType2"] = cellTypesDF.CellType2.values
    RNA.obs["CellType3"] = cellTypesDF.CellType3.values
    RNA.var = featuresGenes
    RNA.var_names = featuresGenes.Gene.values
    RNA.var_names_make_unique()
    if surface:
        RNA = RNA[:, np.isin(RNA.var.values, surfGenes["Gene"].values)]
        RNA.write_h5ad("/opt/CoH/SingleCell/CITE_RNA_Surface.h5ad.gz", compression='gzip')
    else:
        RNA.write_h5ad("/opt/CoH/SingleCell/CITE_RNA.h5ad.gz", compression='gzip')


def importRNACITE(surface=False):
    """Downloads all surface markers and cell types"""
    if surface:
        RNA_DF = ad.read_h5ad("/opt/CoH/SingleCell/CITE_RNA_Surface.h5ad.gz")
    else:
        RNA_DF = ad.read_h5ad("/opt/CoH/SingleCell/CITE_RNA.h5ad.gz")
    return RNA_DF
