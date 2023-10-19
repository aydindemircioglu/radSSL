#!/usr/bin/python3

from scipy.stats import wilcoxon


from sklearn.metrics import roc_curve, auc

from glob import glob
from joblib import Parallel, delayed, load, dump

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tqdm
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    matthews_corrcoef,
    f1_score,
)



from helpers import *
from parameters import *


# for delong
if 1 == 0:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    pROC = importr("pROC")


# def getCI (predsX):
#     Y = predsX["true"].values
#     Y = Y.astype(np.int32)
#     scoresA = predsX["pred"].values
#     lower, auc, upper = pROC.ci(Y, scoresA, direction = "<")
#     return lower, auc, upper
#


def generateSizePlots():
    pts = {}
    for dataID in dList:
        slices = glob(os.path.join(slicesPath, dataID, "*.png"))
        for s in slices:
            pID = s.split("_")[0].split("/")[-1]
            if pID in pts:
                continue
            sp = cv2.imread(s).shape
            pts[pID] = (sp[0], sp[1], dataID)

    if 1 == 1:
        fig, axs = plt.subplots(ncols=1, figsize=(12, 10))
        sns.set_style("whitegrid")
        axs.set_ylabel("Height", fontsize=26)
        axs.set_xlabel("Width", fontsize=26)
        axs.tick_params(axis="x", labelsize=20)
        axs.tick_params(axis="y", labelsize=20)
        g = sns.scatterplot(data=pd.DataFrame(pts).T, x=1, y=0, hue=2)
        x0, x1 = g.get_xlim()
        y0, y1 = g.get_ylim()
        lims = [max(x0, y0), max(x1, y1)]
        plt.setp(axs, xlim=lims, ylim=lims)
        sns.move_legend(axs, "upper left", title="Dataset")
        plt.setp(g.get_legend().get_texts(), fontsize="18")
        plt.setp(g.get_legend().get_title(), fontsize="26")
        g.plot(lims, lims, "-r")
        plt.tight_layout()
        fig = g.get_figure()
        fig.savefig("./results/plot_sizes.png")


def generate3DSizePlots():
    sList = []
    for dataID in dList:
        data = getData(dataID, dropPatID=False, useBlacklist=True, imagePath=radDBPath)

        # we want resampled sizes
        for k in range(len(data)):
            patID = data.iloc[k]["Patient"]
            fvol = glob(os.path.join(cachePath, str(patID) + "*image_1*.nii.gz"))[0]
            fmask = glob(
                os.path.join(cachePath, str(patID) + "*segmentation_1*.nii.gz")
            )[0]

            # just load mask, its faster and they have the same size anyway
            volITK = sitk.ReadImage(fmask)
            mask = sitk.GetArrayFromImage(volITK)
            tmp = np.asarray(mask > 0, dtype=np.uint8)
            zmin, zmax, cmin, cmax, rmin, rmax = getBoundingBox(tmp)  # ITK
            sList.append(
                {
                    "Dataset": dataID,
                    "Patient": patID,
                    "Z": zmax - zmin,
                    "Y": cmax - cmin,
                    "X": rmax - rmin,
                }
            )
    sList = pd.DataFrame(sList)
    sList = sList.sort_values(["X", "Y", "Z"])
    sList.to_excel("./results/sizes_3d.xlsx")

    if 1 == 0:
        fig, axs = plt.subplots(ncols=1, figsize=(12, 10))
        sns.set_style("whitegrid")
        axs.set_ylabel("Height", fontsize=26)
        axs.set_xlabel("Width", fontsize=26)
        axs.tick_params(axis="x", labelsize=20)
        axs.tick_params(axis="y", labelsize=20)
        g = sns.scatterplot(data=pd.DataFrame(pts).T, x=1, y=0, hue=2)
        x0, x1 = g.get_xlim()
        y0, y1 = g.get_ylim()
        lims = [max(x0, y0), max(x1, y1)]
        plt.setp(axs, xlim=lims, ylim=lims)
        sns.move_legend(axs, "upper left", title="Dataset")
        plt.setp(g.get_legend().get_texts(), fontsize="18")
        plt.setp(g.get_legend().get_title(), fontsize="26")
        g.plot(lims, lims, "-r")
        plt.tight_layout()
        fig = g.get_figure()
        fig.savefig("./results/plot_3d_sizes.png")


def getAUCTable():
    AUCTable = {}
    for dataID in dList:
        AUCTable[dataID] = []

        cacheFile = os.path.join("./results/auc_" + dataID + ".dump")
        AUCTable[dataID] = load(cacheFile)
    return AUCTable


def computeAUCTable(dataID):
    cacheFile = os.path.join("./results/auc_" + dataID + ".dump")
    if os.path.exists(cacheFile) == True:
        return None
    AUCTable = {}
    AUCTable[dataID] = []

    print("### Processing", dataID)
    expList = load(f"{resultsPath}/{dataID}_experiments.dump")
    _, fsDict = getExperiments(deepParameters, radParameters)

    expKeys = set(expList["ExpKey"])
    for fsKey in tqdm.tqdm(fsDict.keys()):
        fsTable = []
        for expKey in expKeys:
            subdf = expList.query("Key == @fsKey and ExpKey == @expKey")
            if len(subdf) != 10:
                continue
            assert len(subdf) == 10
            results = []
            for j in range(len(subdf)):
                e = subdf.iloc[j]
                ePath = os.path.join(resultsPath, dataID, str(e["Key"]))
                fResults = os.path.join(
                    ePath, f'{e["Key"]}_{e["ExpKey"]}_{e["Repeat"]}_{e["Fold"]}.dump'
                )
                stats = load(fResults)
                results.append(stats["preds"])

            rf = pd.concat(results)
            assert len(rf) == len(set(rf["Patient"]))
            fpr, tpr, thresholds = roc_curve(
                rf["true"].astype(np.uint32).values, rf["pred"]
            )
            area_under_curve = auc(fpr, tpr)
            fsEntry = subdf.iloc[0].to_dict()
            for _ in ["trainFile", "testFile", "Repeat", "Fold"]:
                fsEntry.pop(_)

            fsEntry["AUC"] = area_under_curve

            # copmute the others too
            true_values = rf["true"].astype(np.uint32).values
            predicted_values = rf["pred"]
            fsEntry["Accuracy"] = accuracy = accuracy_score(
                true_values, predicted_values > 0.5
            )
            fsEntry["Recall"] = recall_score(true_values, predicted_values > 0.5)
            fsEntry["Precision"] = precision_score(true_values, predicted_values > 0.5)
            fsEntry["MCC"] = matthews_corrcoef(true_values, predicted_values > 0.5)
            fsEntry["F1"] = f1_score(true_values, predicted_values > 0.5)

            fsTable.append(fsEntry)
        if len(fsTable) == 0:
            continue
        AUCTable[dataID].append(
            pd.DataFrame(fsTable).sort_values("AUC", ascending=False).iloc[0]
        )

    AUCTable[dataID] = pd.concat(AUCTable[dataID], axis=1).T.reset_index(drop=True)
    dump(AUCTable[dataID], cacheFile)
    return None


def generateMainTable(AUCTable):
    table1 = []
    for dataID in dList:
        cTable = AUCTable[dataID]
        cTable["Model"] = cTable["Model"].replace(np.NaN, "Generic")
        cTable["Force2D"] = cTable["Force2D"].astype(str)
        cTable["Force2D"] = (
            cTable["Force2D"]
            .replace(np.NaN, "")
            .replace("nan", "")
            .replace("True", "+2D")
            .replace("False", "+3D")
        )
        cTable["ModelName"] = (
            cTable["Model"] + "+" + cTable["Fusion"] + cTable["Force2D"]
        )
        cTable[dataID] = cTable["AUC"]
        table1.append(cTable[["ModelName", dataID]])

    for df in table1:
        df.set_index("ModelName", inplace=True)
    table1 = pd.concat(table1, axis=1)
    table1["AUCMean"] = table1.mean(axis=1)
    generic_all_mean = table1.loc["Generic+All+3D", "AUCMean"]

    table1["AUCDiff"] = table1["AUCMean"] - generic_all_mean
    table1.sort_values(["AUCDiff"], ascending=False)

    table1 = table1.sort_values(["AUCDiff"], ascending=False)
    return table1



def generateMetricsTable(tableAll, table1):
    def addToTable (mname, bTable, mTable):
        mAUC = np.round(np.mean(mTable["AUC"]),3)
        mAcc = np.round(np.mean(mTable["Accuracy"]),3)
        mRe = np.round(np.mean(mTable["Recall"]),3)
        mPre = np.round(np.mean(mTable["Precision"]),3)
        mMCC = np.round(np.mean(mTable["MCC"]),3)
        mF1 = np.round(np.mean(mTable["F1"]),3)
        dAUC = np.round(mAUC - rAUC, 3)
        dAcc = np.round(mAcc - rAcc, 3)
        dRe = np.round(mRe - rRe, 3)
        dPre = np.round(mPre - rPre, 3)
        dMCC = np.round(mMCC - rMCC, 3)
        dF1 = np.round(mF1 - rF1, 3)
        bTable.append({"Model": mname, "AUC": mAUC, "AUCDiff": dAUC, "Accuracy": mAcc, "AccuracyDiff": dAcc, "Recall": mRe, "RecallDiff": dRe, "Precision": mPre, "PrecisionDiff": dPre, "MCC": mMCC, "MCCDiff": dMCC, "F1": mF1, "F1Diff": dF1})
        return bTable


    bTable = []

    # extract generic ref values
    tableAll["Force2D"] = tableAll["Force2D"].astype(str)
    gTable = tableAll.query("Model == 'Generic' and Fusion == 'All' and Force2D == '+3D'").copy()
    rAUC = np.round(np.mean(gTable["AUC"]),3)
    rAcc = np.round(np.mean(gTable["Accuracy"]),3)
    rRe = np.round(np.mean(gTable["Recall"]),3)
    rPre = np.round(np.mean(gTable["Precision"]),3)
    rMCC = np.round(np.mean(gTable["MCC"]),3)
    rF1 = np.round(np.mean(gTable["F1"]),3)

    # do this for rad first
    subdf = table1.loc[ [k for k in list(table1.index) if "Generic" in k] ]
    subdf = subdf.sort_values(["AUCMean"], ascending=False).copy()

    for j in range(5):
        mname, f, force2D = subdf.iloc[j].name.split("+")
        force2D = "+"+force2D
        mTable = tableAll.query("Type == 'Generic' and Fusion == @f and Force2D == @force2D").copy()
        assert(mTable.shape[0] == 10)
        bTable = addToTable (subdf.iloc[j].name, bTable, mTable)

    for f in ["None", "Morph", "All"]:
        subdf = table1.loc[ [k for k in list(table1.index) if "+"+f in k and "Generic" not in k] ]
        subdf = subdf.sort_values(["AUCDiff"], ascending=False).copy()

        for j in range(3):
            mname = subdf.iloc[j].name.split("+")[0]
            mTable = tableAll.query("Type == 'Deep' and Fusion == @f and Model == @mname").copy()
            bTable = addToTable (mname, bTable, mTable)
    pd.DataFrame(bTable).to_excel("./paper/Table2.xlsx")
    #return pd.DataFrame(bTable)



def getStr(k):
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


def getName(sList):
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


def generateSupplementalTable1():
    mList = pd.read_excel("./results/modelList.xlsx")
    mList["Model"] = mList["Unnamed: 0"]

    _, fsDict = getExperiments(deepParameters, radParameters)

    modelType = {
        "Medical data": [
            "radimagenet.resnet50",
            "radimagenet.densenet121",
            "radimagenet.inceptionV3",
            "radimagenet.IRV2",
            "medicalimagenet.resnet10",
            "medicalimagenet.resnet18",
            "medicalimagenet.resnet34",
            "medicalimagenet.resnet50",
        ],
        "ImageNet-1K ": [
            "convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k",
            "deit3-huge-p14_in21k-pre_3rdparty_in1k",
            "efficientnet-b7_3rdparty-ra-noisystudent_in1k",
            "efficientnetv2-l_in21k-pre_3rdparty_in1k",
        ],
        "Self-supervised": [
            "simclr_resnet50_16xb256-coslr-200e_in1k",  # 28M params
            "simsiam_resnet50_8xb32-coslr-200e_in1k",  # 38M params
            "mocov3_resnet50_8xb512-amp-coslr-300e_in1k",  # 68M params
            "barlowtwins_resnet50_8xb256-coslr-300e_in1k",
        ],  # 175M params],
        "ImageNet-1K": [
            "resnet34_8xb32_in1k",  # 2.8M params
            "vgg16bn_8xb32_in1k",  # 138M params
            "densenet161_3rdparty_in1k",  # 28M params
            "efficientnet-b2_3rdparty_8xb32_in1k",  # 9.1M params]
        ],
    }

    fTable = []
    for k in modelType:
        for modelID in modelType[k]:
            if "radimage" in modelID or "medical" in modelID:
                p = 0
                f = 0
            else:
                p = mList.query("Model == @modelID").iloc[0]["ParamCount"]
                f = mList.query("Model == @modelID").iloc[0]["Size"]
                f = str(f).replace("torch.Size([", "").replace("])", "").split(",")[0]
            fTable.append(
                {
                    "Model": getStr(modelID),
                    "Pretraining": getStr(k),
                    "#Parameter": p,
                    "#Features": f,
                }
            )
    fTable = pd.DataFrame(fTable)
    fTable.to_excel("./paper/Table_S1.xlsx")


def generateTables(table1):
    table1_generic = table1[table1.index.str.contains("Generic")]
    table1_generic = table1_generic.sort_values(["AUCMean"], ascending=False)
    table1_generic.index = getName(list(table1_generic.index))
    table1_generic = table1_generic.astype(np.float32)
    table1_generic = np.round(table1_generic, 3)
    table1_generic.to_excel("./paper/Table2.xlsx")

    # table 3
    table3_none = table1.loc[
        [k for k in list(table1.index) if "+None" in k and "Generic" not in k]
    ]
    table3_none = table3_none.sort_values(["AUCMean"], ascending=False)
    table3_none = table3_none.astype(np.float32)
    table3_none = np.round(table3_none, 3)
    table3_none.to_excel("./paper/TableS2.xlsx")

    # just the argument that medical pretrained models performed worss, loss >0.057
    table1_med2 = table1[table1.index.str.contains("medical")]

    table4_none = table1.loc[
        [k for k in list(table1.index) if "+Morph" in k and "Generic" not in k]
    ]
    table4_none = table4_none.sort_values(["AUCMean"], ascending=False)
    table4_none = table4_none.astype(np.float32)
    table4_none = np.round(table4_none, 3)
    table4_none.to_excel("./paper/TableS3.xlsx")

    table5_none = table1.loc[
        [k for k in list(table1.index) if "+All" in k and "Generic" not in k]
    ]
    table5_none = table5_none.sort_values(["AUCMean"], ascending=False)
    table5_none = table5_none.astype(np.float32)
    table5_none = np.round(table5_none, 3)
    table5_none.to_excel("./paper/TableS4.xlsx")


def getImprovement(tA, sufA, tB, sufB):
    tC = tA.copy()[dList]
    for k in tA.index:
        subA = tA.query("index == @k")[dList]
        assert len(subA) == 1
        l = k.replace(sufA, sufB)
        subB = tB.query("index == @l")[dList]
        assert len(subB) == 1
        imp = subB[dList].iloc[0] - subA[dList].iloc[0]
        tC.at[k, "Improvement"] = np.mean(imp)
    return tC, np.mean(tC["Improvement"])


def computeSignificance(table1):
    # improvements
    tA = table1.loc[
        [k for k in list(table1.index) if "+None" in k and "Generic" not in k]
    ]
    tB = table1.loc[
        [k for k in list(table1.index) if "+Morph" in k and "Generic" not in k]
    ]
    sufA = "+None"
    sufB = "+Morph"
    tC, x = getImprovement(tA, sufA, tB, sufB)
    _, p = wilcoxon(tC["Improvement"])
    print("Improvement None --> Morph:", x, "p:", p)
    print(np.min(tC["Improvement"]), np.max(tC["Improvement"]))

    tA = table1.loc[
        [k for k in list(table1.index) if "+Morph" in k and "Generic" not in k]
    ]
    tB = table1.loc[
        [k for k in list(table1.index) if "+All" in k and "Generic" not in k]
    ]
    sufA = "+Morph"
    sufB = "+All"
    tC, x = getImprovement(tA, sufA, tB, sufB)
    _, p = wilcoxon(tC["Improvement"])
    print("Improvement Morph --> All:", x, "p:", p)
    print(np.min(tC["Improvement"]), np.max(tC["Improvement"]))

    # just check
    tA = table1.loc[
        [k for k in list(table1.index) if "+None" in k and "Generic" not in k]
    ]
    tB = table1.loc[
        [k for k in list(table1.index) if "+All" in k and "Generic" not in k]
    ]
    sufA = "+None"
    sufB = "+All"
    tC, x = getImprovement(tA, sufA, tB, sufB)
    s, p = wilcoxon(tC["Improvement"])
    print("Improvement None --> All:", x, "p:", p)
    print(np.min(tC["Improvement"]), np.max(tC["Improvement"]))


def compareThem(bestA, preA, bestB, preB):
    _, p = wilcoxon(bestA, bestB)
    print("\n\n#### ")
    print(preA, " mean:", np.mean(bestA))
    print(preB, " mean:", np.mean(bestB))
    d = bestA - bestB
    print("Differences mean:", np.mean(d))
    print("Best", preA, "vs", preB, ":", "p:", p)


def compareModels(table1):
    # compare generic with best deep
    tNone = table1.loc[
        [k for k in list(table1.index) if "+None" in k and "Generic" not in k]
    ]
    bestNone = (
        tNone.sort_values(["AUCMean"], ascending=False)
        .iloc[0][dList]
        .astype(np.float32)
    )

    table1_generic = table1[table1.index.str.contains("Generic")]
    stdGeneric = table1_generic.loc["Generic+All+3D"][dList].astype(np.float32)

    tMorph = table1.loc[
        [k for k in list(table1.index) if "+Morph" in k and "Generic" not in k]
    ]
    bestMorph = (
        tMorph.sort_values(["AUCMean"], ascending=False)
        .iloc[0][dList]
        .astype(np.float32)
    )

    tAll = table1.loc[
        [k for k in list(table1.index) if "+All" in k and "Generic" not in k]
    ]
    bestAll = (
        tAll.sort_values(["AUCMean"], ascending=False).iloc[0][dList].astype(np.float32)
    )
    np.mean(np.round(stdGeneric, 4))

    compareThem(bestNone, "none", stdGeneric, "generic")
    compareThem(bestMorph, "morph", stdGeneric, "generic")
    compareThem(bestAll, "all", stdGeneric, "generic")

    compareThem(bestMorph, "morph", bestNone, "none")

    compareThem(bestAll, "all", bestMorph, "morph")
    compareThem(bestAll, "all", bestNone, "none")



def generateMetricPlots (tableAll):
    measurements = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']

    # extract generic ref values
    tableAll["Force2D"] = tableAll["Force2D"].astype(str)
    gTable = tableAll.query("Model == 'Generic' and Fusion == 'All' and Force2D == '+3D'").copy()
    for m in measurements:
        gTable[m+"_Height"] = 0.01

    for f in ["None", "Morph", "All"]:
        if f == "None":
            strf = "out fusion"
            fname = "./paper/Figure4.png"
        elif f == "Morph":
            strf = " morphological features"
            fname = "./paper/Figure5.png"
        elif f == "All":
            strf = " all hand-crafted features"
            fname = "./paper/Figure6.png"

        filtered_df = tableAll[(tableAll['Type'] == 'Deep') & (tableAll['Fusion'] == f)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        #fig.suptitle("Performance metrics for deep models with"+strf, fontsize=20)
        for i, ax in enumerate(axes.flat):
            measurement = measurements[i]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(filtered_df)))
            sns.scatterplot(x='DataID', y=measurement, legend=False, ax=ax, data=filtered_df)
            gTable.set_index('DataID')[[measurement,measurement+"_Height"]].plot(kind='bar',
                width = 0.9, stacked=True, color=['none', 'red'], ax = ax, legend = False)

            ax.tick_params(axis='x', labelrotation=33, labelsize=13, bottom = 0.5)
            ax.tick_params(axis='y', labelsize=13)
            plt.draw()
            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels, rotation=30, ha='right')
            ax.set_xlabel(None)
            ax.set_ylabel('')
            ax.set_title(measurement, fontsize=21)


            if measurement != "MCC":
                ax.set_ylim([0, 1])
            else:
                ax.set_ylim([-0.15, 1])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.36, wspace=0.33)
        plt.savefig(fname, dpi=333)
        plt.close('all')



def getGTs (row, d):
    expKey = row["ExpKey"]
    fsKey = row["Key"]
    expList = load(f"{resultsPath}/{d}_experiments.dump")
    # get 10 folds
    subdf = expList.query("ExpKey == @expKey and Key == @fsKey")
    assert len(subdf) == 10
    results = []
    for j in range(len(subdf)):
        e = subdf.iloc[j]
        ePath = os.path.join(resultsPath, d, str(e["Key"]))
        fResults = os.path.join(
            ePath, f'{e["Key"]}_{e["ExpKey"]}_{e["Repeat"]}_{e["Fold"]}.dump'
        )
        stats = load(fResults)
        results.append(stats["preds"])
    rf = pd.concat(results)
    gt = rf["true"].values.astype(np.uint32)
    preds = rf["pred"].values.astype(np.float32)
    return gt, preds



def getBestAUCPlotdata (tableAll):
    pdata = {}

    # extract generic ref values
    tableAll["Force2D"] = tableAll["Force2D"].astype(str)
    gTable = tableAll.query("Model == 'Generic' and Fusion == 'All' and Force2D == '+3D'").copy()

    for i, d in enumerate(dList):
        pdata[d] = {}
        GTs = {}
        Preds = {}
        GTs["Hand-Crafted"], Preds["Hand-Crafted"] = getGTs (gTable.query("DataID == @d").iloc[0], d)

        for f in ["None", "Morph", "All"]:
            if f == "None":
                strf = "out fusion"
            elif f == "Morph":
                strf = " morphological features"
            elif f == "All":
                strf = " all hand-crafted features"

            filtered_df = tableAll.query(' Type == "Deep" and Fusion == @f and DataID == @d ')
            filtered_df = filtered_df.sort_values(["AUC"], ascending = False)

            GTs[f], Preds[f] = getGTs (filtered_df.iloc[0], d)
        pdata[d]["GT"] = GTs.copy()
        pdata[d]["Preds"] = Preds.copy()
    return pdata



def generateBestAUCPlots (pdata, psize, fsize):
    fig, axes = plt.subplots(psize[0], psize[1], figsize=fsize)
    for i, ax in enumerate(axes.flat):
        try:
            d = dList[i]
        except:
            # delete subplot
            fig.delaxes(ax)
            continue

        # plot
        colors = ["orange", "red", "darkred", "black"]

        # Iterate over the ground truth and prediction pairs
        for j, pkey in enumerate(["None", "Morph", "All", "Hand-Crafted"]):
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(pdata[d]["GT"][pkey], pdata[d]["Preds"][pkey])
            roc_auc = auc(fpr, tpr)

            # Smooth the ROC curve using moving average
            window_size = 5  # Adjust the window size as needed

            # Linear interpolation for smoother curve
            # fpr_smooth = np.convolve(fpr, np.ones(window_size)/window_size, mode='same')
            # tpr_smooth = np.convolve(tpr, np.ones(window_size)/window_size, mode='same')
            # fpr = fpr_smooth
            # tpr = tpr_smooth

            skey = {"Hand-Crafted": "Hand-crafted", "None": "Deep", "Morph": "Deep+morph", "All": "Deep+hand-crafted"}[pkey]
            #ls = {"Hand-Crafted": "-", "None": "-", "Morph": "--", "All": "-"}[pkey]
            ls = "-"
            # Plot ROC curve
            ax.plot(fpr, tpr, color=colors[j], lw=1.7, linestyle = ls, label=f'{skey} = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('1 - Specificity')
            ax.set_ylabel('Sensitivity')
            ax.set_title(d)
            ax.legend(loc="lower right")

            ax.tick_params(axis='x', labelsize=13)
            ax.tick_params(axis='y', labelsize=13)


    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.36, wspace=0.33)
    plt.savefig("./paper/Figure_S1.png", dpi=333)
    plt.close('all')


def getModelAUCPlotdata (AUCTable, table1):
    pdata = {}

    # extract generic ref values
    tableAll["Force2D"] = tableAll["Force2D"].astype(str)
    gTable = tableAll.query("Model == 'Generic' and Fusion == 'All' and Force2D == '+3D'").copy()

    for i, d in enumerate(dList):
        pdata[d] = {}
        pdata[d]["GT"] = {}
        pdata[d]["Preds"] = {}

    # get best model for All, Morph, None
    for m in ["+All", "+Morph", "+None"]:
        tB = table1.loc[ [k for k in list(table1.index) if m in k and "Generic" not in k] ]
        row = tB.sort_values(["AUCMean"]).iloc[-1]
        model = row.name
        print (f"Best for {m} is {model}")
        for i, d in enumerate(dList):
            pdata[d]["GT"]["Hand-Crafted"], pdata[d]["Preds"]["Hand-Crafted"] = getGTs (gTable.query("DataID == @d").iloc[0], d)

            arow = AUCTable[d].query("ModelName == @model").iloc[0]
            pdata[d]["GT"][model], pdata[d]["Preds"][model] = getGTs (arow, d)
    return pdata



def generateModelAUCPlots (pdata, psize, fsize):
    fig, axes = plt.subplots(psize[0], psize[1], figsize=fsize)
    for i, ax in enumerate(axes.flat):
        try:
            d = dList[i]
        except:
            # delete subplot
            fig.delaxes(ax)
            continue

        # plot

        for j, pkey in enumerate(pdata[d]["GT"].keys()):
            fpr, tpr, _ = roc_curve(pdata[d]["GT"][pkey], pdata[d]["Preds"][pkey])
            roc_auc = auc(fpr, tpr)

            skey = "FIXME"
            if "+All" in pkey:
                skey = "Best deep+handcrafted"
                color = "darkred"
            if "+None" in pkey:
                skey = "Best deep"
                color = "orange"
            if "+Morph" in pkey:
                skey = "Best deep+morph"
                color = "red"
            if "Hand-C" in pkey:
                skey = "Hand-crafted"
                color = "black"

            ls = "-"
            ax.plot(fpr, tpr, color=color, lw=1.7, linestyle = ls, label=f'{skey} = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('1 - Specificity')
            ax.set_ylabel('Sensitivity')
            ax.set_title(d)
            ax.legend(loc="lower right")

            ax.tick_params(axis='x', labelsize=13)
            ax.tick_params(axis='y', labelsize=13)


    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.36, wspace=0.33)
    plt.savefig("./paper/Figure3.png", dpi=333)
    plt.close('all')



def generateTable1():
    df = pd.read_excel("./results/Data.xlsx")
    for i, (idx, row) in enumerate(df.iterrows()):
        d = row["data"]
        data = getData (d, dropPatID = True, useBlacklist = True, imagePath = radDBPath)
        Np = int(np.sum(data["Target"] == 1))
        Nm = int(np.sum(data["Target"] == 0))
        assert(Np + Nm == len(data))
        df.at[idx, "Minor_class"] = Nm if Nm < Np else Np
        df.at[idx, "Major_class"] = Nm if Nm > Np else Np
    df["Minor_class"] = df["Minor_class"].astype(np.uint32)
    df["Major_class"] = df["Major_class"].astype(np.uint32)
    df["Balancedness"] = np.round(df["Major_class"]/df["Minor_class"],2)
    df.to_excel("./paper/Table1.xlsx")



if __name__ == "__main__":
    print("Hi.")

    # some stats
    print("Generating stats")
    generateTable1 ()

    generateSupplementalTable1()

    # obtain results
    print("Generating results")
    fv = Parallel(n_jobs=ncpus)(delayed(computeAUCTable)(dataID) for dataID in dList)

    AUCTable = getAUCTable()
    table1 = generateMainTable(AUCTable)

    tableAll = pd.concat(AUCTable).reset_index(drop = True)
    generateMetricsTable(tableAll, table1)
    generateMetricPlots(tableAll)

    # pdata = getBestAUCPlotdata (tableAll)
    # generateBestAUCPlots(pdata, (4,3), (15,20))

    pdata = getModelAUCPlotdata (AUCTable, table1)
    generateModelAUCPlots(pdata, (4,3), (15,20))

    generateTables(table1)

    computeSignificance(table1)
    compareModels(table1)

#
