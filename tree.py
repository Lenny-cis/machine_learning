# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:27:59 2018

@author: Lenny
"""

from math import log
import pandas as pd


def calcshannoEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
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

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcshannoEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/len(dataSet)
            newEntropy += prob * calcshannoEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

myDat1, myDat2, labels = createDataSet()
%timeit chooseBestFeatureToSplit(myDat1)
%timeit calcshannoEnt(myDat)



























