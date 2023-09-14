#!/usr/bin/python3

import copy
import cv2
import hashlib
import itertools
import json
import logging
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import random
import shutil
import socket
import sys
import tempfile
import time
from tqdm import trange, tqdm

from collections import OrderedDict
from datetime import datetime
from functools import partial
from glob import glob
from joblib import Parallel, delayed, load, dump
from pprint import pprint
from matplotlib import pyplot
from typing import Dict, Any

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import f_classif
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

#from ITMO_FS.filters.univariate import anova

from parameters import *
from helpers import *


# https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/statistical_based/t_score.py
def t_score(X, y):
    n_samples, n_features = X.shape
    F = np.zeros(n_features)
    c = np.unique(y)
    if len(c) == 2:
        for i in range(n_features):
            f = X[:, i]
            # class0 contains instances belonging to the first class
            # class1 contains instances belonging to the second class
            class0 = f[y == c[0]]
            class1 = f[y == c[1]]
            mean0 = np.mean(class0)
            mean1 = np.mean(class1)
            std0 = np.std(class0)
            std1 = np.std(class1)
            n0 = len(class0)
            n1 = len(class1)
            t = mean0 - mean1
            t0 = np.true_divide(std0**2, n0)
            t1 = np.true_divide(std1**2, n1)
            F[i] = np.true_divide(t, (t0 + t1)**0.5)
    else:
        print('y should be guaranteed to a binary class vector')
        exit(0)
    return np.abs(F)



#    wie CV: alle parameter gehen einmal durch
def getMLExperiments (experimentList, expParameters, sKey, inject = None):
    newList = []
    for exp in experimentList:
        for cmb in list(itertools.product(*expParameters.values())):
            pcmb = dict(zip(expParameters.keys(), cmb))
            if inject is not None:
                pcmb.update(inject)
            _exp = exp.copy()
            _exp.append((sKey, pcmb))
            newList.append(_exp)
    experimentList = newList.copy()
    return experimentList



# this is pretty non-generic, maybe there is a better way, for now it works.
def generateAllMLExperiments (experimentParameters, verbose = False):
    experimentList = [ [] ]
    for k in experimentParameters.keys():
        if verbose == True:
            print ("Adding", k)
        elif k == "FeatureSelection":
            # this is for each N too
            print ("Adding feature selection")
            newList = []
            for n in experimentParameters[k]["N"]:
                for m in experimentParameters[k]["Methods"]:
                    fmethod = experimentParameters[k]["Methods"][m].copy()
                    fmethod["nFeatures"] = [n]
                    newList.extend(getMLExperiments (experimentList, fmethod, m))
            experimentList = newList.copy()
        elif k == "Classification":
            newList = []
            for m in experimentParameters[k]["Methods"]:
                newList.extend(getMLExperiments (experimentList, experimentParameters[k]["Methods"][m], m))
            experimentList = newList.copy()
        else:
            experimentList = getMLExperiments (experimentList, experimentParameters[k], k)

    return experimentList



def preprocessData (X, y, simp = None, sscal = None):
    if simp is None:
        simp = SimpleImputer(strategy="mean")
        X = pd.DataFrame(simp.fit_transform(X),columns = X.columns)
    else:
        X = pd.DataFrame(simp.transform(X),columns = X.columns)

    if sscal is None:
        sscal = StandardScaler()
        X = pd.DataFrame(sscal.fit_transform(X),columns = X.columns)
    else:
        X = pd.DataFrame(sscal.transform(X),columns = X.columns)

    return X, y, simp, sscal


# from ITMO.FS
def anova(x, y):
    split_by_class = [x[y == k] for k in np.unique(y)]
    num_classes = len(np.unique(y))
    num_samples = x.shape[0]
    num_samples_by_class = [s.shape[0] for s in split_by_class]
    sq_sum_all = sum((s ** 2).sum(axis=0) for s in split_by_class)
    sum_group = [np.asarray(s.sum(axis=0)) for s in split_by_class]
    sq_sum_combined = sum(sum_group) ** 2
    sum_sq_group = [np.asarray((s ** 2).sum(axis=0)) for s in split_by_class]
    sq_sum_group = [s ** 2 for s in sum_group]
    sq_sum_total = sq_sum_all - sq_sum_combined / float(num_samples)
    sq_sum_within = sum(
        [sum_sq_group[i] - sq_sum_group[i] / num_samples_by_class[i] for i in
         range(num_classes)])
    sq_sum_between = sq_sum_total - sq_sum_within
    deg_free_between = num_classes - 1
    deg_free_within = num_samples - num_classes
    ms_between = sq_sum_between / float(deg_free_between)
    ms_within = sq_sum_within / float(deg_free_within)
    f = ms_between / ms_within
    return np.array(f)


def bhattacharyya_score_fct (X, y):
    yn = y/np.sum(y)
    yn = np.asarray(yn, dtype = np.float32)
    scores = [0]*X.shape[1]
    for j in range(X.shape[1]):
        xn = (X[:,j] - np.min(X[:,j]))/(np.max(X[:,j] - np.min(X[:,j])))
        xn = xn/np.sum(xn)
        xn = np.asarray(xn, dtype = np.float32)
        scores[j] = cv2.compareHist(xn, yn, cv2.HISTCMP_BHATTACHARYYA)

    scores = np.asarray(scores, dtype = np.float32)
    # ties = {i:list(scores).count(i) for i in scores if list(scores).count(i) > 1}
    # print(ties)
    return -scores



def createFSel (fExp, maxFeatures = 4096):
    method = fExp[0][0]
    nFeatures = fExp[0][1]["nFeatures"]
    nFeatures = np.min([nFeatures, maxFeatures])
    pipe = None

    if method == "LASSO":
        C = fExp[0][1]["C"]
        clf = LogisticRegression(penalty='l1', max_iter=500, solver='liblinear', C = C)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures)


    if method == "Anova":
        pipe = SelectKBest(anova, k = nFeatures)


    if method == "ET":
        clf = ExtraTreesClassifier(random_state = 42)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures, threshold=-np.inf)


    if method == "RF":
        clf = RandomForestClassifier(random_state = 42)
        pipe = SelectFromModel(clf, prefit=False, max_features=nFeatures, threshold=-np.inf)


    if method == "Bhattacharyya":
        pipe = SelectKBest(bhattacharyya_score_fct, k = nFeatures)


    if method == "tScore":
        def t_score_fct (X, y):
            scores = t_score (X,y)
            return scores
        pipe = SelectKBest(t_score_fct, k = nFeatures)


    if pipe is None:
        raise Exception ("Method", method, "is unknown")
    return pipe



def createClf (cExp, x_shape):
    #print (cExp)
    method = cExp[0][0]

    if method == "RBFSVM":
        C = cExp[0][1]["C"]
        g = cExp[0][1]["gamma"]
        model = SVC(kernel = "rbf", C = C, gamma = g, probability = True)

    if method == "LogisticRegression":
        C = cExp[0][1]["C"]
        model = LogisticRegression(solver = 'liblinear', C = C, random_state = 42)

    if method == "LinearSVM":
        alpha = cExp[0][1]["alpha"]
        model = SGDClassifier(alpha = alpha, loss = "log")

    if method == "RandomForest":
        n_estimators = cExp[0][1]["n_estimators"]
        #max_features = cExp[0][1]["max_features"]
        model = RandomForestClassifier(n_estimators = n_estimators)#, max_features = max_features)

    if method == "TabPFNClassifier":
        model = TabPFNClassifier(device='cpu')

    if method == "NaiveBayes":
        model = GaussianNB()

    if method == "NeuralNetwork":
        N1 = cExp[0][1]["layer_1"]
        N2 = cExp[0][1]["layer_2"]
        N3 = cExp[0][1]["layer_3"]
        model = MLPClassifier (hidden_layer_sizes=(N1,N2,N3,), random_state=42, max_iter = 50)
    return model

    executeExperiment (fselExperiments, clfExperiments, traindf, testdf, eConfig)



@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=UserWarning)
def executeExperiment (traindf, testdf, eConfig, fResults):
    print ("X", end = '', flush = True)
    fExp = eConfig["FSel"]
    cExp = eConfig["Clf"]

    stats = {}
    np.random.seed(42)
    random.seed(42)

    stats = {}
    stats["features"] = []
    stats["Ntrain"] = traindf.shape[0]
    stats["Ntest"] = testdf.shape[0]
    stats["params"] = {}
    stats["params"].update(fExp)
    stats["params"].update(cExp)

    # need a fixed set of folds to be comparable
    timeFSStart = time.time()

    X_train = traindf.copy()
    y_train = X_train["Target"]
    X_train = X_train.drop(["Target", "Patient", "Image", "mask"], axis = 1)

    X_test = testdf.copy()
    y_test = X_test["Target"]
    testIDs = X_test["Patient"]
    X_test = X_test.drop(["Target", "Patient", "Image", "mask"], axis = 1)

    # make sure we have something numeric, at least for mrmre
    y_train = y_train.astype(np.uint8)

    # scale
    X_train, y_train, simp, sscal = preprocessData (X_train, y_train)
    X_test, y_test, _, _ = preprocessData (X_test, y_test, simp, sscal)

    # create fsel
    fselector = createFSel (fExp, maxFeatures = X_train.shape[1])
    with np.errstate(divide='ignore',invalid='ignore'):
        fselector.fit (X_train.copy(), y_train.copy())
    feature_idx = fselector.get_support()
    feature_names = X_train.columns[feature_idx].copy()
    stats["features"].append(list([feature_names][0].values))

    # apply selector-- now the data is numpy, not pandas, lost its names
    X_fs_train = fselector.transform (X_train)
    y_fs_train = y_train

    X_fs_test = fselector.transform (X_test)
    y_fs_test = y_test

    # check if we have any features
    if X_fs_train.shape[1] > 0:
        classifier = createClf (cExp, X_fs_train.shape)
        classifier.fit (np.array(X_fs_train, dtype = np.float32), np.array(y_fs_train, dtype = np.int64))

        y_pred = classifier.predict_proba (np.array(X_fs_test, dtype = np.float32))
        assert(classifier.classes_ == [0,1]).all()
        y_pred = y_pred[:,1]
    else:
        # else we can just take 1 as a prediction
        y_pred = y_test*0 + 1
    tmpY = pd.DataFrame([y_pred, testIDs, y_test], index=["pred", "Patient", "true"]).T
    tmpY["pred"] = tmpY["pred"].astype(np.float32)
    stats["preds"] = tmpY

    # time
    timeFSEnd = time.time()
    stats["Time_Overall"] =  timeFSEnd - timeFSStart
    stats["eConfig"] = eConfig
    dump (stats, fResults)
    pass


def executeExperiments (e, traindf, testdf):
    ePath = os.path.join(resultsPath, dataID, str(e["Key"]))
    fResults = os.path.join(ePath, f'{e["Key"]}_{e["ExpKey"]}_{e["Repeat"]}_{e["Fold"]}.dump')
    if os.path.exists(fResults):
        print (".", end = '', flush = True)
        return None

    # ensure it exists
    os.makedirs (ePath, exist_ok = True)
    executeExperiment (traindf, testdf, e, fResults)



if __name__ == "__main__":
    print ("Hi.")
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)

    # generate all experiments
    fselExperiments = generateAllMLExperiments (fselParameters)
    print ("Created", len(fselExperiments), "feature selection parameter settings")
    clfExperiments = generateAllMLExperiments (clfParameters)
    print ("Created", len(clfExperiments), "classifier parameter settings")
    print ("Total", len(clfExperiments)*len(fselExperiments), "experiments")


    set_random_seed(42, deterministic=True)
    _, fsDict = getExperiments (deepParameters, radParameters)

    for dataID in dList:
        print ("Processing", dataID)

        fResults = os.path.join(resultsPath, dataID)
        os.makedirs (fResults, exist_ok = True)

        expList = []
        dataCache = {}
        fPath = os.path.join(featuresPath, dataID)
        print ("Loading datasets and creating configs")
        for fsKey in tqdm(fsDict):
            trainFiles = glob(f"{fPath}/{fsKey}_{dataID}_*_train.csv")
            for trainFile in trainFiles:
                testFile = trainFile.replace("_train", "_test")
                dataCache[trainFile] = pd.read_csv(trainFile)
                dataCache[testFile] = pd.read_csv(testFile)

                # generate list of experiment combinations
                fsConfig = fsDict[fsKey]
                fsConfig["Key"] = fsKey
                fsConfig["trainFile"] = trainFile
                fsConfig["testFile"] = testFile

                for fe in fselExperiments:
                    for clf in clfExperiments:
                        eConfig = {"FSel": fe, "Clf": clf}
                        expKey = dict_hash(eConfig)
                        fConfig = fsConfig.copy()
                        fConfig.update(eConfig)
                        fConfig["ExpKey"] = expKey
                        fConfig["DataID"] = dataID
                        # extract repeat and fold as well
                        r = int(os.path.basename(fConfig["trainFile"]).split("_")[2])
                        f = int(os.path.basename(fConfig["trainFile"]).split("_")[3])
                        fConfig["Repeat"] = r
                        fConfig["Fold"] = f
                        expList.append( fConfig)

        # just dump for evaluation
        dump(pd.DataFrame(expList), f"{resultsPath}/{dataID}_experiments.dump")

        # execute
        print ("Executing training")
        fv = Parallel (n_jobs = ncpus)(delayed(executeExperiments)(e, traindf = dataCache[e["trainFile"]], testdf = dataCache[e["testFile"]]) for e in expList)



#
