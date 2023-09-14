from collections import OrderedDict
import itertools
import json
import numpy as np
import os

from typing import Dict, Any


basedir = basePath = "/data/data/radSSL"
radDBPath = "/data/radDatabase"


# train parameters
num_workers = 16
aggregationType = "max"

patchSize = 224

checkpoint_dir = os.path.join(basedir, "checkpoint")

### parameters
nRepeats = 1
nCV = 10
ncpus = 30

dList = [ "Desmoid", "Lipo", "Liver", "Melanoma", "GIST", "GBM", "CRLM", "HN", "C4KCKiTS", "ISPY1" ]
dList = sorted(dList)

CT_datasets = ['HN', 'GIST', 'CRLM', 'Melanoma', 'C4KCKiTS']
MR_datasets = ['Lipo', 'Desmoid', 'Liver', 'ISPY1', 'GBM']

featuresPath = os.path.join(basePath, "features")
cachePath = os.path.join(basePath, "cache")
resultsPath = os.path.join(basePath, "results")
slicesPath = os.path.join(basedir, "slices")
checkpointsPath = os.path.join(basedir, "checkpoints")


preprocessParameters = {"Resampling" : [1]}


# if we use more binwidth, we would need to combine it
# with all levels, which would take too much time
radParameters = OrderedDict({
    "BinWidth": [25],
    "Fusion": ["Morph", "All", "NoMorph"],
    "Force2D": [True, False]
})


deepParameters = OrderedDict({
    # these are 'one-of'
    "Model": [  # RadImageNet
                "radimagenet.resnet50",
                "radimagenet.densenet121",
                "radimagenet.inceptionV3",
                "radimagenet.IRV2",

                # MedicalImageNet
                "medicalimagenet.resnet10",
                "medicalimagenet.resnet18",
                "medicalimagenet.resnet34",
                "medicalimagenet.resnet50",

                # Best performing models
                "convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k",
                "deit3-huge-p14_in21k-pre_3rdparty_in1k",
                "efficientnet-b7_3rdparty-ra-noisystudent_in1k",
                # "beit-base-p16_beitv2-in21k-pre_3rdparty_in1k",  # crashed somehow
                "efficientnetv2-l_in21k-pre_3rdparty_in1k",

                # Self-supervised models
                "simclr_resnet50_16xb256-coslr-200e_in1k", # 28M params
                "simsiam_resnet50_8xb32-coslr-200e_in1k", # 38M params
                "mocov3_resnet50_8xb512-amp-coslr-300e_in1k", # 68M params
                "barlowtwins_resnet50_8xb256-coslr-300e_in1k", # 175M params

                # ImageNet pretrained models, standard training without fancy anything
                "resnet34_8xb32_in1k", #2.8M params
                "vgg16bn_8xb32_in1k", #138M params
                "densenet161_3rdparty_in1k", #28M params
                "efficientnet-b2_3rdparty_8xb32_in1k" #9.1M params

                ],
    "Fusion": ["None", "Morph", "All"]
})


fselParameters = OrderedDict({
    # these are 'one-of'
    "FeatureSelection": {
        "N": [1,2,4,8,16,32,64,128],
        "Methods": {
            "LASSO": {"C": [1.0]},
            "Anova": {},
            "tScore": {},
            "Bhattacharyya": {},
            "RF": {}
        }
    }
})



clfParameters = OrderedDict({
    "Classification": {
        "Methods": {
            "LogisticRegression": {"C": np.logspace(-8, 8, 5, base = 2.0) },
            "NeuralNetwork": {"layer_1": [16, 64, 256], "layer_2": [16, 64, 256], "layer_3": [16, 64, 256]},
            "RandomForest": {"n_estimators": [50,125,250]},
        }
    }
})



#
