#
import pandas as pd
from joblib import Parallel, delayed
import os
import multiprocessing
import time
import multiprocessing.pool
import functools
from glob import glob
import nibabel as nib
import nibabel.processing as nibp

from parameters import *
from helpers import *


def processFile (row, p):
    f = row["Image"]
    fmask = row["mask"]

    fout_img = getFout (f, resampling = p, outPath = cachePath)
    fout_mask = getFout (fmask, resampling = p, outPath = cachePath)

    # USELESS, since it is reacreated!!
    if os.path.exists(fout_img) and os.path.exists(fout_mask):
        return None

    img = nib.load(f)
    seg = nib.load(fmask)


    # make sure the mask is all 0 and 1
    tmp = seg.get_fdata()
    tmp = np.asarray(tmp > 0, dtype = np.uint8)

    # copy also over affine infos, else we can end up with volumes that are SLIGHTLY different
    new_seg = seg.__class__(tmp, img.affine, img.header)
    rimg = nibp.resample_to_output(img, voxel_sizes = [p, p, p], order = 3)
    rSeg = nibp.resample_to_output(new_seg, voxel_sizes = [p, p, p], order = 0)

    test = np.median(rSeg.get_fdata())
    assert (test == 0.0 or test == 1.0)

    #print (f, rimg.shape, rSeg.shape)
    assert (rimg.shape == rSeg.shape)
    print ("X", end = '', flush = True)

    nib.save(rimg, fout_img)
    nib.save(rSeg, fout_mask)
    pass



if __name__ == '__main__':
    recreatePath (cachePath)
    for d in dList:
        data = getData (d, dropPatID = True, useBlacklist = True, imagePath = radDBPath)
        for p in preprocessParameters ["Resampling"]:
            fv = Parallel (n_jobs = ncpus)(delayed(processFile)(row, p) for (idx, row) in data.iterrows())

#
