import os
import numpy as np
from scipy.special import binom
import cv2
import itertools
import random
import pandas as pd
from glob import glob
from pprint import pprint
import json
import hashlib
from typing import Dict, Any
from math import sqrt
import shutil

from skimage.measure import label

#from parameters import *


# from mmdet
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def dict_hash(dictionary: Dict[str, Any]) -> str:
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()



def getExperiments (deepParameters, radParameters):
    # create deep sets
    expList = [dict(zip(deepParameters.keys(), z) ) for z in list(itertools.product(*deepParameters.values()))]
    for v in range(len(expList)):
        expList[v]["Type"] = "Deep"

    # create generic sets
    rexpList = [dict(zip(radParameters.keys(), z) ) for z in list(itertools.product(*radParameters.values()))]
    for v in range(len(rexpList)):
        rexpList[v]["Type"] = "Generic"

    expList.extend(rexpList)

    expDict = {}
    for e in expList:
        dname = dict_hash(e)
        expDict[dname] = e

    return expList, expDict



def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    largestCC = 255*largestCC
    return np.asarray(largestCC, dtype = np.uint8)



def recreatePath (path, create = True):
    print ("Recreating path ", path)
    try:
        shutil.rmtree (path)
    except:
        pass

    if create == True:
        try:
            os.makedirs (path)
        except:
            pass
    print ("Done.")



def getFout (f, resampling = 1, outPath = None):
    if "/HN" in f:
        f = f.replace("mask_GTV-1", "segmentation")
        patID = [pt for pt in f.split("/") if "HN" in pt]
        assert (len(patID) == 1)
        pdir = patID[0].split("_")[0]
        fout = os.path.join(pdir + "_" + os.path.basename(f).replace(".nii.gz", "_" + str(resampling) + ".nii.gz"))
        fout = os.path.join(outPath,  fout)
    else:
        pdir = (os.path.basename(os.path.dirname(f)))
        fout = os.path.join(pdir + "_" + os.path.basename(f).replace(".nii.gz", "_" + str(resampling) + ".nii.gz"))
        fout = os.path.join(outPath,  fout)

    # fix for KiTS
    if "KiTS-" in f:
        fout = fout.replace("_2_1.", "_segmentation_1.")
        fout = fout.replace("Image_", "image_")
    # stupid
    if "GBM-" in f or "ISPY1-" in f:
        fout = fout.replace("_Mask_", "_segmentation_")
        fout = fout.replace("_Image_", "_image_")
    if "Melanoma-" in f:
        fout = fout.replace("_lesion0", "")
    if "KiTS-" in f:
        fout = fout.replace("_2_1.", "_segmentation_1.")
        fout = fout.replace("Image_", "image_")
    # always apply, but for CRLM and GIST-018
    fout = fout.replace("_lesion_0", "")
    fout = fout.replace("_lesion0_RAD", "")
    fout = fout.replace("_Lipoma", "")
    return fout



def findOptimalCutoff (fpr, tpr, threshold, verbose = False):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    fpr, tpr, threshold

    Returns
    -------
    list type, with optimal cutoff value

    """

    # own way
    minDistance = 2
    bestPoint = (2,-1)
    for i in range(len(fpr)):
        p = (fpr[i], tpr[i])
        d = sqrt ( (p[0] - 0)**2 + (p[1] - 1)**2 )
        if verbose == True:
            print (p, d)
        if d < minDistance:
            minDistance = d
            bestPoint = p

    if verbose == True:
        print ("BEST")
        print (minDistance)
        print (bestPoint)
    sensitivity = bestPoint[1]
    specificity = 1 - bestPoint[0]
    return sensitivity, specificity



def dict_hash(dictionary: Dict[str, Any]) -> str:
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()




# blacklist comes from another project,
# contains either those not processable by pyradiomics
# or those with too large slice thickness
def getData (dataID, dropPatID = False, useBlacklist = True, imagePath = None):
    # load data first
    data = pd.read_csv("./data/pinfo_" + dataID + ".csv")
    if useBlacklist == True:
        blacklist = pd.read_csv("./data/blacklist.csv").T.values[0]
        data = data.query("Patient not in @blacklist").copy()

    # add path to data
    for i, (idx, row) in enumerate(data.iterrows()):
        image, mask = getImageAndMask (dataID, row["Patient"], imagePath)
        data.at[idx, "Image"] = image
        data.at[idx, "mask"] = mask
    #print ("### Data shape", data.shape)

    data["Target"] = data["Diagnosis"]
    data = data.drop(["Diagnosis"], axis = 1).reset_index(drop = True).copy()
    if dropPatID == True:
        data = data.drop(["Patient"], axis = 1).reset_index(drop = True).copy()

    # make sure we shuffle it and shuffle it the same
    np.random.seed(111)
    random.seed(111)
    data = data.sample(frac=1)

    return data



def getImageAndMask (d, patName, imagePath = None):
    image = os.path.join(imagePath, patName, "image.nii.gz")
    if d in ["HN"]:
        # nifti only exists with CT, so PET will be ignored
        cands = glob(os.path.join(imagePath, patName + "*/**/image.nii.gz"), recursive = True)
        if len(cands) != 1:
            print ("Error with ", patName)
            print ("Checked", os.path.join(imagePath, patName + "*/**/image.nii.gz"))
            pprint(cands)
            raise Exception ("Cannot find image.")
        image = cands[0]
        cands = glob(os.path.join(imagePath, patName + "*/**/mask_GTV-1.nii.gz"), recursive = True)
        if len(cands) != 1:
            print ("Error with ", patName)
            pprint(cands)
            raise Exception ("Cannot find mask.")
        mask = cands[0]
    if d in ["Desmoid", "GIST", "Lipo", "Liver"]:
        mask = os.path.join(imagePath, patName, "segmentation.nii.gz")
    if d in ["CRLM"]:
        mask = os.path.join(imagePath, patName, "segmentation_lesion0_RAD.nii.gz")
    if d in ["Melanoma"]:
        mask = os.path.join(imagePath, patName, "segmentation_lesion0.nii.gz")


    if d in ["ISPY1", "GBM"]:
        # 2 is the renal tumor
        image = os.path.join(imagePath, patName, "Image.nii.gz")
        mask = os.path.join(imagePath, patName, "Mask.nii.gz")

    if d in ["C4KCKiTS"]:
        # 2 is the renal tumor
        image = os.path.join(imagePath, patName, "Image.nii.gz")
        mask = os.path.join(imagePath, patName, "2.nii.gz")


    # special cases
    if patName == "GIST-018":
        image = os.path.join(imagePath, patName, "image_lesion_0.nii.gz")
        mask = os.path.join(imagePath, patName, "segmentation_lesion_0.nii.gz")
    if patName == "Lipo-073":
        mask = os.path.join(imagePath, patName, "segmentation_Lipoma.nii.gz")

    if os.path.exists(image) == False:
        print ("Missing", image)
    if os.path.exists(mask) == False:
        print ("Missing", mask)
    return image, mask




# https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def getBoundingBox(img, expFactor = 0.1):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    rmin = np.max([int(rmin - (rmax-rmin)*expFactor) , 0])
    cmin = np.max([int(cmin - (cmax-cmin)*expFactor), 0])
    rmax = np.min([int(rmax + (rmax - rmin)*expFactor), img.shape[0]])
    cmax = np.min([int(cmax + (cmax-cmin)*expFactor), img.shape[1]])

    return rmin, rmax, cmin, cmax, zmin, zmax


##


bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=2000):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 1000)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points.
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)


# https://stackoverflow.com/questions/50731785/create-random-shape-contour-using-matplotlib
# returns a mask with label 255, channel 0 = mask, channel 1 = image
def generateMask (img, cseed, patchSize):
    h, w = img.shape
    mask = np.zeros((h,w), dtype = np.uint8)

    np.random.seed(cseed)
    random.seed(cseed)
    edgy = random.uniform(0,2)
    rad = random.uniform(0,2)
    n = random.randint(3,16)

    pointsOK = False
    while pointsOK == False:
        try:
            # midpoint
            c_h = np.random.normal(0,1)*h//8+h//2
            c_w = np.random.normal(0,1)*w//8+w//2

            sc = np.min([c_h,c_w,h-c_h, w-c_w])
            sc = random.randint(16,np.min([sc,patchSize]))

            a = get_random_points(n=n, scale=sc)
            a =a + (c_h,c_w)

            x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)

            # first try coordinates, if something is not ok, this will throw
            for z in zip(*(y,x)):
                k = mask[int(z[1]), int(z[0])]
            for z in zip(*(y,x)):
                mask[int(z[1]), int(z[0])] = 255
            pointsOK = True
        except:
            pass

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contourIdx, cnt in enumerate(contours):
        # compute a convex hull
        hull = cv2.convexHull(cnt)

        # fill the inside with red
        cv2.fillPoly(mask, pts=[hull], color=(255))

    # is the patch too large?
    pa0 = np.where(np.sum(mask, axis = 1) > 0)[0]
    pa1 = np.where(np.sum(mask, axis = 0) > 0)[0]
    ph = np.max(pa0)-np.min(pa0)
    pw = np.max(pa1)-np.min(pa1)
    if ph > patchSize:
        ph = patchSize
        pa0 = pa0[0:patchSize+1]

    if pw > patchSize:
        pw = patchSize
        # just cut the patch
        pa1 = pa1[0:patchSize+1]

    # extract patch
    maskpatch = np.zeros((patchSize,patchSize,1), dtype = np.float32)
    pofsh = (patchSize - ph)//2
    pofsw = (patchSize - pw)//2
    maskpatch[pofsh:pofsh+ph, pofsw:pofsw+pw, 0] = mask[np.min(pa0):np.max(pa0), np.min(pa1):np.max(pa1)]

    imgpatch = np.zeros((patchSize,patchSize,1), dtype = img.dtype)
    imgpatch[pofsh:pofsh+ph, pofsw:pofsw+pw, 0] = img[np.min(pa0):np.max(pa0), np.min(pa1):np.max(pa1)]
    return maskpatch, imgpatch




#
