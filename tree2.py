# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 21:04:56 2018

@author: Lenny
"""

from math import log
import pandas as pd
import numpy as np


def calcshannoEnt(Group):
    numEntries = Group.sum()
    labelCounts = dict(Group.groupby(level=0).sum())
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    A = np.random.randint(0, 2, 100000)
    B = np.random.randint(0, 2, 100000)
    F = ['yes' if x == 1 else 'no' for x in np.random.randint(0, 2, 100000)]
    dataSet1 = pd.DataFrame({'A':A, 'B':B, 'F':F}).values.tolist()
    dataSet2 = pd.DataFrame({'A':A, 'B':B, 'F':F})
    labels = ['no surfacing', 'flippers']
    return dataSet1, dataSet2, labels


def splitDataSet(Group, featureid, value):
    return Group.groupby(level=[0, featureid]).sum().loc[:, value]


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet.index.names) - 1
    baseEntropy = calcshannoEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        uniqueVals = set(dataSet.index.levels[i])
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = subDataSet.sum()/len(dataSet)
            newEntropy += prob * calcshannoEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

myDat1, myDat2, labels = createDataSet()
g = myDat2.groupby(['F', 'A', 'B']).size()
%timeit chooseBestFeatureToSplit(g)





















