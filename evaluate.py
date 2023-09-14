#!/usr/bin/python3

from collections import OrderedDict
from datetime import datetime
from scipy.stats import wilcoxon

from contextlib import contextmanager
from sklearn.calibration import calibration_curve
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve

from glob import glob
from joblib import dump, load
from matplotlib import cm
from matplotlib import pyplot
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.transforms import Bbox
from PIL import Image
from PIL import ImageDraw, ImageFont
from scipy.stats import spearmanr, pearsonr
from typing import Dict, Any
import copy
import cv2
import hashlib
import itertools
import json
import math
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import tqdm
import pathlib
import pickle
import pylab
import random
import scipy.cluster.hierarchy as sch
import seaborn as sns
import shutil
import sys
import tempfile
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from scipy import stats

from helpers import *
from parameters import *


# for delong
if 1 == 0:
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    pROC = importr('pROC')


def getCI (predsX):
    Y = predsX["true"].values
    Y = Y.astype(np.int32)
    scoresA = predsX["pred"].values
    lower, auc, upper = pROC.ci(Y, scoresA, direction = "<")
    return lower, auc, upper



def generateSizePlots ():
    pts = {}
    for dataID in dList:
        slices = glob (os.path.join(slicesPath, dataID, "*.png"))
        for s in slices:
            pID = s.split("_")[0].split("/")[-1]
            if pID in pts:
                continue
            sp = cv2.imread(s).shape
            pts[pID] = (sp[0], sp[1], dataID)

    if 1 == 1:
        fig, axs = plt.subplots(ncols=1, figsize = (12,10))
        sns.set_style("whitegrid")
        axs.set_ylabel("Height", fontsize= 26)
        axs.set_xlabel("Width", fontsize= 26)
        axs.tick_params(axis='x', labelsize=20 )
        axs.tick_params(axis='y', labelsize=20 )
        g = sns.scatterplot(data = pd.DataFrame(pts).T , x = 1, y = 0, hue = 2)
        x0, x1 = g.get_xlim()
        y0, y1 = g.get_ylim()
        lims = [max(x0, y0), max(x1, y1)]
        plt.setp(axs, xlim=lims, ylim=lims)
        sns.move_legend(axs, "upper left",  title = "Dataset")
        plt.setp(g.get_legend().get_texts(), fontsize='18')
        plt.setp(g.get_legend().get_title(), fontsize='26')
        g.plot(lims, lims, '-r')
        plt.tight_layout()
        fig = g.get_figure()
        fig.savefig("./results/plot_sizes.png")




def generate3DSizePlots ():
    sList = []
    for dataID in dList:
        data = getData (dataID, dropPatID = False, useBlacklist = True, imagePath = radDBPath)

        # we want resampled sizes
        for k in range(len(data)):
            patID = data.iloc[k]["Patient"]
            fvol = glob(os.path.join(cachePath, str(patID) +"*image_1*.nii.gz") )[0]
            fmask = glob(os.path.join(cachePath, str(patID) +"*segmentation_1*.nii.gz") )[0]

            # just load mask, its faster and they have the same size anyway
            volITK = sitk.ReadImage(fmask)
            mask = sitk.GetArrayFromImage(volITK)
            tmp = np.asarray(mask > 0, dtype = np.uint8)
            zmin, zmax, cmin, cmax, rmin, rmax = getBoundingBox(tmp) # ITK
            sList.append({"Dataset": dataID, "Patient": patID, "Z": zmax - zmin, "Y": cmax - cmin, "X": rmax -rmin})
    sList = pd.DataFrame(sList)
    sList = sList.sort_values(["X", "Y", "Z"])
    sList.to_excel("./results/sizes_3d.xlsx")

    if 1 == 0:
        fig, axs = plt.subplots(ncols=1, figsize = (12,10))
        sns.set_style("whitegrid")
        axs.set_ylabel("Height", fontsize= 26)
        axs.set_xlabel("Width", fontsize= 26)
        axs.tick_params(axis='x', labelsize=20 )
        axs.tick_params(axis='y', labelsize=20 )
        g = sns.scatterplot(data = pd.DataFrame(pts).T , x = 1, y = 0, hue = 2)
        x0, x1 = g.get_xlim()
        y0, y1 = g.get_ylim()
        lims = [max(x0, y0), max(x1, y1)]
        plt.setp(axs, xlim=lims, ylim=lims)
        sns.move_legend(axs, "upper left",  title = "Dataset")
        plt.setp(g.get_legend().get_texts(), fontsize='18')
        plt.setp(g.get_legend().get_title(), fontsize='26')
        g.plot(lims, lims, '-r')
        plt.tight_layout()
        fig = g.get_figure()
        fig.savefig("./results/plot_3d_sizes.png")




def getAUCTable ():
    AUCTable = {}
    for dataID in dList:
        AUCTable[dataID] = []

        cacheFile = os.path.join("./results/auc_"+ dataID + ".dump")
        if os.path.exists (cacheFile) == True:
            AUCTable[dataID] = load(cacheFile)
            continue

        print ("### Processing", dataID)
        expList = load(f"{resultsPath}/{dataID}_experiments.dump")
        _, fsDict = getExperiments (deepParameters, radParameters)

        expKeys = set(expList["ExpKey"])
        for fsKey in tqdm.tqdm(fsDict.keys()):
            fsTable = []
            for expKey in expKeys:
                subdf = expList.query("Key == @fsKey and ExpKey == @expKey")
                if len(subdf) != 10:
                    continue
                assert(len(subdf) == 10)
                results = []
                for j in range(len(subdf)):
                    e = subdf.iloc[j]
                    ePath = os.path.join(resultsPath, dataID, str(e["Key"]))
                    fResults = os.path.join(ePath, f'{e["Key"]}_{e["ExpKey"]}_{e["Repeat"]}_{e["Fold"]}.dump')
                    stats = load (fResults)
                    results.append(stats["preds"])

                rf = pd.concat(results)
                assert(len(rf) == len(set(rf["Patient"])))
                fpr, tpr, thresholds = roc_curve (rf["true"].astype(np.uint32).values, rf["pred"])
                area_under_curve = auc (fpr, tpr)
                fsEntry = subdf.iloc[0].to_dict()
                for _ in ["trainFile", "testFile", "Repeat", "Fold"]:
                    fsEntry.pop(_)

                fsEntry["AUC"] = area_under_curve

                l95, area_under_curve, u95 = getCI(rf)
                fsEntry["CI_low"] = l95
                fsEntry["CI_high"] = u95
                ci95 = str(np.round(area_under_curve,2)) + " (" + str(np.round(l95,2)) + "-" + str(np.round(u95,2)) + ")"
                fsEntry["CI"] = ci95

                fsTable.append(fsEntry)
            if len(fsTable) == 0:
                continue
            AUCTable[dataID].append(pd.DataFrame(fsTable).sort_values("AUC", ascending = False).iloc[0])

        AUCTable[dataID] = pd.concat(AUCTable[dataID], axis = 1).T.reset_index(drop = True)
        dump(AUCTable[dataID], cacheFile)
        #AUCTable[dataID].to_csv(f"./results/AUCTable_{dataID}.csv")
    return AUCTable



def generateMainTable (AUCTable):
    table1 = []
    for dataID in dList:
        cTable = AUCTable[dataID]
        cTable["Model"] = cTable["Model"].replace(np.NaN, "Generic")
        cTable["Force2D"] = cTable["Force2D"].astype(str)
        cTable["Force2D"] = cTable["Force2D"].replace(np.NaN, "").replace("nan","").replace("True", "+2D").replace("False", "+3D")
        cTable["ModelName"] = cTable["Model"] + "+" + cTable["Fusion"] + cTable["Force2D"]
        cTable[dataID] = cTable["AUC"]
        table1.append(cTable[["ModelName", dataID]])

    for df in table1:
        df.set_index("ModelName", inplace=True)
    table1 = pd.concat(table1, axis=1)
    table1['AUCMean'] = table1.mean(axis=1)
    generic_all_mean = table1.loc['Generic+All+3D', 'AUCMean']

    table1['AUCDiff'] = table1['AUCMean'] - generic_all_mean
    table1.sort_values(["AUCDiff"], ascending = False)

    table1 = table1.sort_values(["AUCDiff"], ascending = False)
    return table1




def getPerDatasetPlot (DPI = 200):
    bP = []
    for d in dList:
        bP.append({"Dataset": d, "Type": "Generic", "AUC": np.max(table1_generic[d])})
        bP.append({"Dataset": d, "Type": "Medical", "AUC":np.max(table1_med[d])})
        bP.append({"Dataset": d, "Type": "ImageNet", "AUC":np.max(table1_other[d])})
    bP = pd.DataFrame(bP)
    o = bP.query("Type == 'ImageNet'").sort_values(["AUC"])["Dataset"]
    DPI = 70
    fig, ax = plt.subplots(1,3, figsize = (20, 6), dpi = DPI)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('AUC', fontsize = 22, labelpad = 12)
    plt.xlabel('Dataset', fontsize= 22, labelpad = 12)

    for j, subkey in enumerate(["None", "Morph", "All"]):
        gmkeys = [k for k in list(table1_med.index) if "+"+subkey in k]
        table1_med_none = table1_med.query("index in @gmkeys")
        gmkeys = [k for k in list(table1_other.index) if "+"+subkey in k]
        table1_other_none = table1_other.query("index in @gmkeys")
        bP = []
        for d in dList:
            bP.append({"Dataset": d, "Type": "Generic", "AUC": np.max(table1_generic[d])})
            bP.append({"Dataset": d, "Type": "Medical", "AUC":np.max(table1_med_none[d])})
            bP.append({"Dataset": d, "Type": "ImageNet", "AUC":np.max(table1_other_none[d])})

        if 1 == 1:
            cp = sns.color_palette("hls", 3)
            bP = pd.DataFrame(bP)
            bP = bP.set_index('Dataset').loc[o].reset_index()
            sns.lineplot(ax = ax[j], x = "Dataset", y = "AUC", hue = "Type", data = bP, palette = cp, sort = False, linewidth = 3)

            plt.setp(ax[j].get_legend().get_texts(), fontsize='16') # for legend text
            plt.setp(ax[j].get_legend().get_title(), fontsize='20') # for legend title
            #ax.set_xticks(nList[1:])#, rotation = 0, ha = "right", fontsize = 22)
            ax[j].xaxis.tick_bottom()
            ax[j].yaxis.tick_left()

        plt.tight_layout()
        fig.savefig("./results/Figure_4.png", facecolor = 'w', bbox_inches='tight')




def getStr (k):
    if k == "Generic+NoMorph+2D":
        return "Generic, -Morph, 2D"
    if k == "Generic+All+2D":
        return "Generic, 2D"
    if k == "Generic+All+3D":
        return "Generic, 3D"
    if k == "Generic+Morph+2D":
        return "Generic, Morph, 2D"
    if k == "Generic+Morph+3D":
        return "Generic, Morph, 3D"
    if k == "Generic+NoMorph+3D":
        return "Generic, -Morph, 3D"
    return k



def getName (sList):
    oList = []
    for k in sList:
        oList.append(getStr(k))
    return oList



def find_key_with_value(dictionary, X):
    result = []
    for key, value in dictionary.items():
        if X in value:
            result.append(key)
    return result



def generateSupplementalTable1 ():
    mList = pd.read_excel("./results/modelList.xlsx")
    mList["Model"] = mList["Unnamed: 0"]

    _, fsDict = getExperiments (deepParameters, radParameters)

    modelType = {\
            "Medical data": [
                "radimagenet.resnet50",
                "radimagenet.densenet121",
                "radimagenet.inceptionV3",
                "radimagenet.IRV2",
                "medicalimagenet.resnet10",
                "medicalimagenet.resnet18",
                "medicalimagenet.resnet34",
                "medicalimagenet.resnet50"],
            "ImageNet-1K ": [
                "convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k",
                "deit3-huge-p14_in21k-pre_3rdparty_in1k",
                "efficientnet-b7_3rdparty-ra-noisystudent_in1k",
                "efficientnetv2-l_in21k-pre_3rdparty_in1k"],
            "Self-supervised": [
                "simclr_resnet50_16xb256-coslr-200e_in1k", # 28M params
                "simsiam_resnet50_8xb32-coslr-200e_in1k", # 38M params
                "mocov3_resnet50_8xb512-amp-coslr-300e_in1k", # 68M params
                "barlowtwins_resnet50_8xb256-coslr-300e_in1k"], # 175M params],
            "ImageNet-1K": [
                "resnet34_8xb32_in1k", #2.8M params
                "vgg16bn_8xb32_in1k", #138M params
                "densenet161_3rdparty_in1k", #28M params
                "efficientnet-b2_3rdparty_8xb32_in1k" #9.1M params]
            ]}


    fTable = []
    for k in modelType:
        for modelID in modelType[k]:
            if "radimage" in modelID or "medical" in modelID:
                p = 0
                f = 0
            else:
                p = mList.query("Model == @modelID").iloc[0]["ParamCount"]
                f = mList.query("Model == @modelID").iloc[0]["Size"]
                f = str(f).replace("torch.Size([", ''). replace("])", '').split(",")[0]
            fTable.append({"Model": getStr(modelID), "Pretraining": getStr(k), "#Parameter": p, "#Features": f})
    fTable = pd.DataFrame(fTable)
    fTable.to_excel("./results/Table_S1.xlsx")



def generateTables (table1):
    table1_generic = table1[table1.index.str.contains("Generic")]
    table1_generic = table1_generic.sort_values(["AUCMean"], ascending = False)
    table1_generic.index = getName(list(table1_generic.index))
    table1_generic = table1_generic.astype(np.float32)
    table1_generic = np.round(table1_generic,3)
    table1_generic.to_excel("./paper/Table2.xlsx")


    # table 3
    table3_none = table1.loc[[k for k in list(table1.index) if "+None" in k and "Generic" not in k]]
    table3_none = table3_none.sort_values(["AUCMean"], ascending = False).iloc[0:10]
    table3_none = table3_none.astype(np.float32)
    table3_none = np.round(table3_none,3)
    table3_none.to_excel("./paper/Table3.xlsx")

    # just the argument that medical pretrained models performed worss, loss >0.057
    table1_med2 = table1[table1.index.str.contains("medical")]


    table4_none = table1.loc[[k for k in list(table1.index) if "+Morph" in k and "Generic" not in k]]
    table4_none = table4_none.sort_values(["AUCMean"], ascending = False).iloc[0:10]
    table4_none = table4_none.astype(np.float32)
    table4_none = np.round(table4_none,3)
    table4_none.to_excel("./paper/Table4.xlsx")


    table5_none = table1.loc[[k for k in list(table1.index) if "+All" in k and "Generic" not in k]]
    table5_none = table5_none.sort_values(["AUCMean"], ascending = False).iloc[0:10]
    table5_none = table5_none.astype(np.float32)
    table5_none = np.round(table5_none,3)
    table5_none.to_excel("./paper/Table5.xlsx")
    pass


def getImprovement(tA, sufA, tB, sufB):
    tC = tA.copy()[dList]
    for k in tA.index:
        subA = tA.query("index == @k")[dList]
        assert (len(subA) == 1)
        l = k.replace(sufA, sufB)
        subB = tB.query("index == @l")[dList]
        assert (len(subB) == 1)
        imp = subB[dList].iloc[0] - subA[dList].iloc[0]
        tC.at[k, "Improvement"] = np.mean(imp)
    return tC, np.mean(tC["Improvement"])


def computeSignificance(table1):
    # improvements
    tA = table1.loc[[k for k in list(table1.index) if "+None" in k and "Generic" not in k]]
    tB = table1.loc[[k for k in list(table1.index) if "+Morph" in k and "Generic" not in k]]
    sufA = "+None"
    sufB = "+Morph"
    tC, x = getImprovement(tA, sufA, tB, sufB)
    _, p = wilcoxon(tC["Improvement"])
    print ("Improvement None --> Morph:", x, "p:", p)
    print (np.min(tC["Improvement"]), np.max(tC["Improvement"]))

    tA = table1.loc[[k for k in list(table1.index) if "+Morph" in k and "Generic" not in k]]
    tB = table1.loc[[k for k in list(table1.index) if "+All" in k and "Generic" not in k]]
    sufA = "+Morph"
    sufB = "+All"
    tC, x = getImprovement(tA, sufA, tB, sufB)
    _, p = wilcoxon(tC["Improvement"])
    print ("Improvement Morph --> All:", x, "p:", p)
    print (np.min(tC["Improvement"]), np.max(tC["Improvement"]))

    # just check
    tA = table1.loc[[k for k in list(table1.index) if "+None" in k and "Generic" not in k]]
    tB = table1.loc[[k for k in list(table1.index) if "+All" in k and "Generic" not in k]]
    sufA = "+None"
    sufB = "+All"
    tC, x = getImprovement(tA, sufA, tB, sufB)
    s, p = wilcoxon(tC["Improvement"])
    print ("Improvement None --> All:", x, "p:", p)
    print (np.min(tC["Improvement"]), np.max(tC["Improvement"]))



def compareThem(bestA, preA, bestB, preB):
    _, p = wilcoxon(bestA, bestB)
    print ("\n\n#### ")
    print (preA," mean:", np.mean(bestA))
    print (preB," mean:", np.mean(bestB))
    d = bestA - bestB
    print ("Differences mean:", np.mean(d))
    print ("Best",preA,"vs",preB, ":", "p:", p)
    pass


def compareModels(table1):
    # compare generic with best deep
    tNone = table1.loc[[k for k in list(table1.index) if "+None" in k and "Generic" not in k]]
    bestNone = tNone.sort_values(["AUCMean"], ascending = False).iloc[0][dList].astype(np.float32)

    table1_generic = table1[table1.index.str.contains("Generic")]
    stdGeneric = table1_generic.loc["Generic+All+3D"][dList].astype(np.float32)

    tMorph = table1.loc[[k for k in list(table1.index) if "+Morph" in k and "Generic" not in k]]
    bestMorph = tMorph.sort_values(["AUCMean"], ascending = False).iloc[0][dList].astype(np.float32)

    tAll = table1.loc[[k for k in list(table1.index) if "+All" in k and "Generic" not in k]]
    bestAll = tAll.sort_values(["AUCMean"], ascending = False).iloc[0][dList].astype(np.float32)
    np.mean(np.round(stdGeneric,4))

    compareThem(bestNone, "none", stdGeneric, "generic")
    compareThem(bestMorph, "morph", stdGeneric, "generic")
    compareThem(bestAll, "all", stdGeneric, "generic")

    compareThem(bestMorph, "morph", bestNone, "none")

    compareThem( bestAll, "all", bestMorph, "morph")
    compareThem(bestAll, "all", bestNone, "none")


if __name__ == '__main__':
    print ("Hi.")

    # some stats
    print ("Generating stats")
    #generateSizePlots()
    #generate3DSizePlots()
    generateSupplementalTable1 ()

    # obtain results
    print ("Generating results")

    AUCTable = getAUCTable()
    table1 = generateMainTable (AUCTable)
    generateTables (table1)
    computeSignificance(table1)
    compareModels(table1)

#
