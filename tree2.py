# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 21:04:56 2018

@author: Lenny
"""

from math import log
import pandas as pd
import numpy as np
import operator


def calcshannoEnt(Group):
    numEntries = Group.sum()
    labelCounts = dict(Group.groupby(level=-1).sum())
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    A = np.random.randint(0, 2, 100000)
    B = np.random.randint(0, 2, 100000)
    F = ['yes' if x == 1 else 'no' for x in np.random.randint(0, 2, 100000)]
#    dataSet = [[1, 1, 'yes'],
#           [1, 1, 'yes'],
#           [1, 0, 'no'],
#           [0, 1, 'no'],
#           [0, 1, 'no']]

    dataSet1 = pd.DataFrame(dataSet, columns=['A', 'B', 'F']).values.tolist()
    dataSet2 = pd.DataFrame(dataSet, columns=['A', 'B', 'F'])
    labels = ['no surfacing', 'flippers']
    return dataSet1, dataSet2, labels


def splitDataSet(Group, featureid, value):
    newGroup = Group[Group.index.get_level_values(level=featureid) == value]
    return pd.Series.from_array(newGroup.values,
                                index=newGroup.index.droplevel(
                                        level=featureid))


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet.index.names) - 1
    numEntries = dataSet.sum()
    baseEntropy = calcshannoEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        uniqueVals = set(dataSet.index.levels[i])
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = subDataSet.sum()/numEntries
            newEntropy += prob * calcshannoEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def mayorityCnt(classList):
    return classList.groupby(level=0).sum()\
        .sort_values(ascending=False).index[0]


def createTree(dataSet, labels):
    classList = dataSet.groupby(level=-1).sum()
    if len(classList.index) == 1:
        return classList.index[0]
    if len(dataSet.index.names) == 1:
        return mayorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    uniqueVals = set(dataSet.index.levels[bestFeat])
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(
                dataSet, bestFeat, value), subLabels)
    return myTree


myDat1, myDat2, labels = createDataSet()
g = myDat2.groupby(['A', 'B', 'F']).size()    
%timeit createTree(g, labels[:])





















