#
import os
import pandas as pd
from radiomics import featureextractor
from sklearn.model_selection import RepeatedStratifiedKFold

import json
from joblib import Parallel, delayed, load, dump

# this shit simply hangs if the GPU memory is full. instead of warning
# or erroring. it just stalls. thank you tensorflow, now die.
import tensorflow as tf
from keras import backend as K

# take care of tensorflow shit. fucking die already!!
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)



from helpers import *
from parameters import *

import medicalnet
from mmpretrain import get_model
from KerasFeatureExtractor import KerasFeatureExtractor
from MMPretrainFeatureExtractor import MMPretrainFeatureExtractor
from MedicalImageNetFeatureExtractor import MedicalImageNetFeatureExtractor


def generateSplits ():
    print ("### Generating splits")
    for dataID in dList:
        data = getData (dataID, dropPatID = False, useBlacklist = True, imagePath = radDBPath)

        # check if we have fixed the CV or not
        cacheFile = f"./results/CVSplit_{dataID}.dump"
        if os.path.exists(cacheFile) == False:
            idxList = {}
            for r in range(nRepeats):
                kfolds = RepeatedStratifiedKFold(n_splits = nCV, n_repeats = 1, random_state = 2022)
                idxList[r] = {}
                for k, (train_index, test_index) in enumerate(kfolds.split(data["Patient"], data["Target"])):
                    trainIDs = list(data.iloc[train_index]["Patient"])
                    testIDs = list(data.iloc[test_index]["Patient"])
                    idxList[r][k] = (trainIDs, testIDs)
            dump (idxList, cacheFile)
    print ("    Done.")



def getRadiomicsFV (dataID, row, config):
    # check if we have that
    patID = row["Patient"]
    fvcachePath = os.path.join(featuresPath, dataID, "cache")
    os.makedirs(fvcachePath, exist_ok = True)
    fvcacheFile = os.path.join(fvcachePath, f"{dataID}_{patID}_{config['expKey']}.csv")
    if os.path.exists(fvcacheFile):
        #print ("Found from cache:", cacheFile)
        print ("C", end = '', flush = True)
        fv = pd.read_csv(fvcacheFile)
        return fv

    # need to recompute
    print ("R", end = '', flush = True)
    if dataID in MR_datasets:
        params = os.path.join("config/MR.yaml")
    else:
        params = os.path.join("config/CT.yaml")
    try:
        # copy the only parameter we need
        eParams = {"binWidth":config["BinWidth"], "force2D": config["Force2D"]}
        extractor = featureextractor.RadiomicsFeatureExtractor(params, **eParams)
        if config["Force2D"] == True:
            extractor.enableImageTypeByName("LBP2D")
        else:
            extractor.enableImageTypeByName("LBP3D")
        # use from cache
        fvol = glob(os.path.join(cachePath, str(patID) +"*image_1*.nii.gz") )[0]
        fmask = glob(os.path.join(cachePath, str(patID) +"*segmentation_1*.nii.gz") )[0]

        f = extractor.execute(fvol, fmask, label = 1)

        # we prepare it all so it can be used directly, so remove all diagnostics here
        f = {p:f[p] for p in f if "diagnost" not in p}
        for k in row.keys():
            f[k] = row[k]
        fv = pd.DataFrame(f, index = [0])
        fv.to_csv(fvcacheFile, index = False)
    except Exception as e:
        #f = pd.DataFrame([{"ERROR": patID}])
        print (f"#### GOT AN ERROR for {patID}!", e)
        raise Exception ("really wrong here?")
    return fv



def extractRadiomicFeatures (traindf, dataID, config, removeDiag = True):
    trainFeats = []
    for j in range(len(traindf)):
        fv = getRadiomicsFV (dataID, traindf.iloc[j], config)
        trainFeats.append(fv.iloc[0])
    trainFeats = pd.DataFrame(trainFeats)
    trainFeats.reset_index(drop = True)
    # remove diags--- but its not there right now, maybe turned off
    # during extraction. will change this if they pop up suddenly
    return trainFeats



# will return always Patient, Image, mask and Target, so subsequent tasks can
# work with these
def filterFeats (trainFeats, featType):
    if featType == "All":
        pass
    if featType == "Morph":
        morphKeys = [m for m in trainFeats.keys() if "original_shape" in m]
        morphKeys += ["Patient", "Image", "mask", "Target"]
        trainFeats = trainFeats[morphKeys]
    if featType == "NoMorph":
        nomorphKeys = [m for m in trainFeats.keys() if "original_shape" not in m]
        trainFeats = trainFeats[nomorphKeys]
    if featType == "None":
        baseKeys = ["Patient", "Image", "mask", "Target"]
        trainFeats = trainFeats[baseKeys]
    return trainFeats



def getSlices (row, dataID, config):
    if "medicalimagenet" in config["Model"]:
        return pd.DataFrame(row).T

    patID = row["Patient"]
    allSlices = glob(os.path.join(slicesPath, dataID, f"*{patID}*.png"), recursive = True)
    return allSlices


def extractDeepFeatures (traindf, dataID, config):
    # now add deep features
    # preload model will lead to CUDA error if we run in parallel, using cache
    model = None
    extractor = None

    def transform_image(img):
        # radimagent example has this:
        # image = (image-127.5)*2 / 255
        # but also loads a pytorch model. wtf?
        img = tf.image.resize_with_pad(img, target_height=224, target_width=224, method=tf.image.ResizeMethod.BICUBIC)
        img = (img - 127.5) * 2 / 255.0
        return img


    # no normalization needed since this is part of the preprocessor of each config
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='ResizeEdge', scale=224, edge='short', backend='pillow', interpolation='bicubic'),
        dict(type='CenterCrop', crop_size=224),
        dict(type='PackInputs')
    ]

    deepdf = []
    for j in range(len(traindf)):
        patID = traindf.iloc[j]["Patient"]
        fvcachePath = os.path.join(featuresPath, dataID, "cache")
        os.makedirs(fvcachePath, exist_ok = True)
        # TODO: using model instead of expkey will reduce computations by 3
        #fvcacheFile = os.path.join(fvcachePath, f"{dataID}_{patID}_{config['expKey']}.csv")
        fvmodelcacheFile = os.path.join(fvcachePath, f"{dataID}_{patID}_{config['Model']}.csv")
        # if os.path.exists(fvcacheFile):
        #     #print ("Found from cache:", cacheFile)
        #     print ("E", end = '', flush = True)
        #     slFeats = pd.read_csv(fvcacheFile)["0"]
        #     # fix it
        #     shutil.copyfile (fvcacheFile, fvmodelcacheFile)
        if os.path.exists(fvmodelcacheFile):
            print ("M", end = '', flush = True)
            slFeats = pd.read_csv(fvmodelcacheFile)["0"]
        else:
            print ("N", end = '', flush = True)

            # complex model generation
            if model is None:
                if "radimagenet" in config["Model"]:
                    if config["Model"] == "radimagenet.resnet50":
                        keras_model = f"{basedir}/pretrained/radimagenet/RadImageNet-ResNet50_notop.h5"
                    elif config["Model"] == "radimagenet.densenet121":
                        keras_model = f"{basedir}/pretrained/radimagenet/RadImageNet-DenseNet121_notop.h5"
                    elif config["Model"] == "radimagenet.inceptionV3":
                        keras_model = f"{basedir}/pretrained/radimagenet/RadImageNet-InceptionV3_notop.h5"
                    elif config["Model"] == "radimagenet.IRV2":
                        keras_model = f"{basedir}/pretrained/radimagenet/RadImageNet-IRV2_notop.h5"
                    else:
                        raise Exception ("Unknown RadImageNet model!")
                    model = tf.keras.models.load_model(keras_model)
                    layer_name = model.layers[-1].name
                    extractor = KerasFeatureExtractor(model, layer_name, transform_image)
                elif "medicalimagenet" in config["Model"]:
                    resnetdim = None
                    if config["Model"] == "medicalimagenet.resnet10":
                        mpath = f"{basedir}/pretrained/medicalimagenet/resnet_10_23dataset.pth"
                        shortcutType = "B"
                        resnetdim = 10
                    elif config["Model"] == "medicalimagenet.resnet18":
                        mpath = f"{basedir}/pretrained/medicalimagenet/resnet_18_23dataset.pth"
                        shortcutType = "A"
                        resnetdim = 18
                    elif config["Model"] == "medicalimagenet.resnet34":
                        mpath = f"{basedir}/pretrained/medicalimagenet/resnet_34_23dataset.pth"
                        shortcutType = "A"
                        resnetdim = 34
                    elif config["Model"] == "medicalimagenet.resnet50":
                        mpath = f"{basedir}/pretrained/medicalimagenet/resnet_50_23dataset.pth"
                        shortcutType = "B"
                        resnetdim = 50
                    else:
                        raise Exception ("Unknown MedicalImageNet model!")
                    W = 112; H = 112; D = 56
                    model = medicalnet.generate_model(resnetdim, False, shortcutType, None, W, H, D, mpath).cuda()
                    _ = model.eval()
                    extractor = MedicalImageNetFeatureExtractor(model, D, H, W)
                else:
                    model = get_model(config["Model"], backbone=dict(out_indices=(0, 1, 2, 3)), pretrained=True).cuda()
                    _ = model.eval()
                    model._config.test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
                    extractor = MMPretrainFeatureExtractor(model, stage = "backbone", agg = "max")

            slices = getSlices (traindf.iloc[j], dataID, config)
            slFeats = extractor(slices)
            slFeats.to_csv(fvmodelcacheFile, index = False)

        # aggegrate
        slFeats = {f"feat_{i}":s  for i, s in enumerate(slFeats)}

        # then back add to traindf
        for k in traindf.iloc[j].keys():
            slFeats[k] = traindf.iloc[j][k]
        deepdf.append(slFeats)

    if model is not None:
        K.clear_session()
        del model

    deepdf = pd.DataFrame(deepdf)
    return deepdf



# now we load each split
def processDataset (dataID, expKey, expDict, firstFoldOnly = False):
    print ("### Processing", dataID)

    # load data and splits
    data = getData (dataID, dropPatID = False, useBlacklist = True, imagePath = radDBPath)
    cacheFile = f"./results/CVSplit_{dataID}.dump"
    idxList = load(cacheFile)

    config = expDict[expKey].copy()
    config["expKey"] = expKey
    for r in idxList.keys():
        for f in idxList[r].keys():
            if firstFoldOnly == True:
                if r != 0 or f != 0:
                    continue
            print (f"Repeat {r}, Fold {f}")

            trainFile = f"{expKey}_{dataID}_{r}_{f}_train.csv"
            testFile = f"{expKey}_{dataID}_{r}_{f}_test.csv"
            fPath = os.path.join(featuresPath, dataID)
            os.makedirs(fPath, exist_ok = True)
            trainFile = os.path.join(fPath, trainFile)
            testFile = os.path.join(fPath, testFile)
            if os.path.exists(trainFile) == True:
                print (f"{trainFile} exists.")
                continue

            # get IDs
            trainIDs, testIDs = idxList[r][f]
            traindf = data.query("Patient in @trainIDs")
            testdf = data.query("Patient in @testIDs")

            # if we extract deep, we want to use 3D generic+all,
            # so we overwrite
            rconfig = config.copy()
            if config["Type"] == "Deep":
                rconfig = {k:expDict[k] for k in expDict if expDict[k]["Type"] == "Generic" and expDict[k]["Force2D"] == False and expDict[k]["Fusion"] == "All"}
                assert(len(rconfig) == 1)
                rexpKey = list(rconfig.keys())[0]
                rconfig = rconfig[rexpKey]
                rconfig["expKey"] = rexpKey
            trainFeats = extractRadiomicFeatures (traindf, dataID, rconfig)
            testFeats = extractRadiomicFeatures (testdf, dataID, rconfig)

            trainFeats = filterFeats (trainFeats, config["Fusion"])
            testFeats = filterFeats (testFeats, config["Fusion"])

            if config["Type"] == "Deep":
                trainFeats = extractDeepFeatures (trainFeats, dataID, config)
                testFeats = extractDeepFeatures (testFeats, dataID, config)

            trainFeats.to_csv(trainFile, index = False)
            testFeats.to_csv(testFile, index = False)



if __name__ == '__main__':
    print ("Hi.")
    print ("Generating splits.")
    os.makedirs ("./results", exist_ok = True)
    generateSplits()

    if torch.cuda.is_available() == False:
        raise Exception ("No graphics card?")

    set_random_seed(42, deterministic=True)
    _, expDict = getExperiments (deepParameters, radParameters)

    # dump those just for reference
    os.makedirs ("./results/fsetconfigs/", exist_ok = True)
    for k in expDict:
        j = json.dumps(expDict[k], indent=4)
        with open('./results/fsetconfigs/'+str(k)+".json", 'w') as f:
            print(j, file= f)

    # because of fusion we first filter the radiomic datasets
    # this we can run in parallel, no GPU involved
    print ("### Preparing generic features")
    radDict = {k:expDict[k] for k in expDict if expDict[k]["Type"] == "Generic"}
    _ = Parallel (n_jobs = ncpus)(delayed(processDataset)(dataID, expKey, expDict) for expKey in radDict for dataID in dList)

    # then do rest (radiomics is there, so wont be recomputed )
    # only compute first fold, the rest will then be computed from cached features
    print ("### Extracting deep features")
    for expKey in expDict:
        print (expKey)
        print (expDict[expKey])
        for dataID in dList:
            processDataset (dataID, expKey, expDict, firstFoldOnly = True)

    # then we can assemble the rest in parallel, no gpu used
    _ = Parallel (n_jobs = ncpus)(delayed(processDataset)(dataID, expKey, expDict, firstFoldOnly = False) for expKey in expDict for dataID in dList)

#
