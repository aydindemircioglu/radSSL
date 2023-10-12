#
import json

import pandas as pd
import logging
import numpy as np
import SimpleITK as sitk

from helpers import *
from parameters import *


def gatherInfos(blackList, dataID, metas):
    rx, ry, rz = [], [], []
    sx, sy, sz = [], [], []
    data = getData (dataID, dropPatID = False, useBlacklist = False, imagePath = radDBPath)

    metas[dataID] = {}
    for i, (idx, row) in enumerate(data.iterrows()):
        if row["Patient"] in blackList:
            continue
        reader = sitk.ImageFileReader()
        reader.SetFileName(row["Image"])
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        rx.append(float(reader.GetMetaData("pixdim[1]")))
        ry.append(float(reader.GetMetaData("pixdim[2]")))
        rz.append(float(reader.GetMetaData("pixdim[3]")))
        sx.append(float(reader.GetMetaData("dim[1]")))
        sy.append(float(reader.GetMetaData("dim[2]")))
        sz.append(float(reader.GetMetaData("dim[3]")))
        metas[dataID][row["Patient"]] = rz[-1]
    return rx, ry, rz, sx, sy, sz, metas


def getStr (rx):
    s = str(np.round(np.median(rx), 1)) + " (" + str( np.round(np.min(rx),1) ) + " - " + str(np.round(np.max(rx),1)) + ")"
    return s



if __name__ == "__main__":
    print ("Hi.")

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    blackList = []
    metas = {}
    for dataID in dList:
        _, _, _, _, _, _, metas = gatherInfos(blackList, dataID, metas)

    # generate blackList
    for dataID in dList:
        data = getData (dataID, dropPatID = False, useBlacklist = False, imagePath = radDBPath)
        rmedian = np.median(list(metas[dataID].values()))
        #print (dataID, rmedian)
        for k in metas[dataID]:
            if metas[dataID][k] > 2*rmedian:
                blackList.append(k)

    # write blacklist
    pd.DataFrame(blackList).to_csv("./data/blacklist.csv", index = False)

    # gather without those
    dTable = []
    for dataID in dList:
        rx, ry, rz, sx, sy, sz, metas = gatherInfos(blackList, dataID, metas)
        if dataID in CT_datasets:
            mod = "CT"
        else:
            mod = "MR"
        dTable.append({"data": dataID, "Modality": mod, "N": len(rx), "Inplane Resolution": getStr(rx),  "Slice Thickness": getStr(rz)})
        #print(dataID, sorted(rz))
    dTable = pd.DataFrame(dTable)
    dTable.to_excel("./results/Data.xlsx")
    print (dTable)

#
