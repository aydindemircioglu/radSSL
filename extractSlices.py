#
import random
import pandas as pd
import SimpleITK as sitk
from glob import glob
import json
import seaborn as sns
import cv2
import nibabel as nib
import argparse
import logging
import math
import numpy as np
import itertools
from pathlib import Path
from joblib import Parallel, delayed, load, dump


from helpers import *
from parameters import *
from joblib import Parallel, delayed



def extractSlices (dataset, dataID):
    data = dataset.copy()
    finalData = []

    for i, (idx, row) in enumerate(data.iterrows()):
        fvol = glob(os.path.join(cachePath, row["Patient"] +"*image_1*.nii.gz") )
        fmask = glob(os.path.join(cachePath, row["Patient"] +"*segmentation_1*.nii.gz") )
        print (fvol, fmask, row["Patient"])
        assert(len(fvol) == 1)
        assert(len(fmask) == 1)

        volITK = sitk.ReadImage(fvol)
        volMaskITK = sitk.ReadImage(fmask)
        vol = sitk.GetArrayFromImage(volITK)[0,:,:,:]
        volMask = sitk.GetArrayFromImage(volMaskITK)[0,:,:,:]

        print ("X", end = '', flush = True)
        if volMask.shape != vol.shape:
            print (data, row)
            assert(volMask.shape == vol.shape)

        # find mask volume
        # rmin, rmax, cmin, cmax, zmin, zmax = getBoundingBox(vol) # nibable
        zmin, zmax, cmin, cmax, rmin, rmax = getBoundingBox(volMask) # ITK
        if rmax <= rmin or cmax <= cmin or zmax <= zmin:
            print (rmin, rmax, cmin, cmax)
            print (row)
            raise Exception("Something is wrong with the bounding box")
        vol = vol[:, cmin:cmax, rmin:rmax]
        volMask = volMask[:, cmin:cmax, rmin:rmax]

        # change intensities globally (=slab..) here, not per slice
        if dataID in CT_datasets:
            vol[vol < -1024] = -1024
            vol[vol > 2048] = 2048
            vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))
            vol = np.asarray(255*vol, dtype = np.uint8)
        else:
            vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))
            vol = np.asarray(255*vol, dtype = np.uint8)


        # identify slices that we can use
        slices = []
        for h in range(volMask.shape[0]):
            # append slice if area is more than 50 mm2 = 7mm*7mm + eps
            a = np.sum(np.abs(volMask[h,:,:]))
            if a > 50:
                slices.append( [h, a] )
        if len(slices) == 0:
            raise Exception (str(row)+ "\nhas empty slices, data:" + str(row))
            continue
        slices.sort(key=lambda x: x[1])
        eSlices = slices

        for s in eSlices:
            img = vol[s[0],:,:].copy()
            mask = volMask[s[0],:,:].copy()
            os.makedirs( os.path.join(slicesPath, dataID), exist_ok = True)
            imgName = os.path.join(slicesPath, dataID, f'{row["Patient"]}_{s[0]}_image.nii.gz')
            maskName = os.path.join(slicesPath, dataID, f'{row["Patient"]}_{s[0]}_mask.nii.gz')

            test = np.median(mask)
            assert (test == 0.0 or test == 1.0 or test == 0.5)
            assert (img.shape == mask.shape)
            print ("X", end = '', flush = True)

            # mask needs to be resized to 255 for ROIchannel/ROIcut
            mask = mask*255.0

            aImg = np.stack((img,)*3, axis=-1)
            aImg[:,:,1] = mask
            aImg[:,:,2] = mask*0.5 + aImg[:,:,0]*0.5
            aImg = np.asarray(aImg, dtype = np.uint8)
            #aImg = np.asarray(aImg*255, dtype = np.uint8)

            # image contains mask
            pngimgName = os.path.join(slicesPath, dataID, f'{row["Patient"]}_{s[0]}_image.png')
            cv2.imwrite(pngimgName, aImg)

    return finalData



def extractAllSlices (dataID):
    print ("### Processing", dataID)
    data = getData (dataID, dropPatID = False, useBlacklist = True, imagePath = radDBPath)
    extractSlices (data, dataID)
    return None

 

if __name__ == "__main__":
    print ("Hi.")

    # create paths
    os.makedirs( slicesPath, exist_ok = True)
    _ = Parallel (n_jobs = 10)(delayed(extractAllSlices)(dataID) for dataID in dList)

#
