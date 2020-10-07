
import math
import pandas as pd

class Node:
    def __init__(self, isLeaf, label, threshold, gainRatio=None):
        self.isLeaf = isLeaf
        self.label = label
        self.threshold = threshold
        self.gainRatio = gainRatio
        self.children = []

#Decision Tree Implementation for C4.5 algorithm
#TODO- this is only for two child nodes
class DecisionTree:
    def __init__(self, classes, maxDepth=None):
        self.maxDepth = maxDepth
        self.classes = classes

    def fit(self, xData, yData):
        columns = xData.columns.tolist()
        self.rootNode = self.generateTree(xData, yData, columns)

    # # First learn a decision tree classifier to boost the learning process
    # dt = DecisionTreeClassifier(criterion=learningCriteria, random_state=100,
    #                             max_depth=maxLevels, min_samples_leaf=minSamplesPerLeaf)
    # dt.fit(xData, yData)

    def generateTree(self, xData, yData, columns):
        (best, best_threshold, splitted) = self.bestSplit(xData, yData, columns)
        print("Best", best)
        print("best thresh", best_threshold)
        print("Splitted", splitted)

        remainingAttributes = columns[:]
        remainingAttributes.remove(best)
        node = Node(isLeaf=False, label=best, threshold=best_threshold)
        node.children = [self.bestSplit(subset, yData, remainingAttributes) for subset in splitted]

        return node

    def getFeatureType(self, ft, data):
        values = set(data[ft])
        if len(values) < 10:
            return "Discrete"
        else:
            return "Continuous"



    def bestSplit(self, xData, yData, features):
        maxEntropy = -999999
        bestFt = None

        # None for discrete attributes, threshold value for continuous attributes
        bestThreshold = None
        for ft in features:
            print("ft:", ft)

            if self.getFeatureType(ft, xData) == 'Discrete': #if is discrete value (can only take specific values)
                # split curData into n-subsets, where n is the number of
                # different values of attribute i. Choose the attribute with
                # the max gain
                discreteVals = list(set(xData[ft]))
                subsets = [pd.DataFrame() for a in discreteVals]
                for row in xData:
                    for index in range(len(discreteVals)):
                        if row[ft] == discreteVals[index]:
                            subsets[index] = subsets[index].append(row)
                            # break
                e = self.gainRatio(xData, yData, subsets)
                if e > maxEntropy:
                    maxEntropy = e
                    splitted = subsets
                    bestFt = ft
                    bestThreshold = None
            else:
                # sort the data according to the column.Then try all
                # possible adjacent pairs. Choose the one that
                # yields maximum gain
                # data.sort(key=lambda x: x[ftIndex])
                # sort the data according to the column.Then try all
                # possible adjacent pairs. Choose the one that
                # yields maximum gain
                # curData.sort(key=lambda x: x[indexOfAttribute])

                for j in range(len(xData.index)-1):
                    idx1 = xData.index[j]
                    idx2 = xData.index[j+1]

                    if xData.loc[idx1][ft] != xData.loc[idx2][ft]:
                        threshold = (xData.loc[idx1][ft] + xData.loc[idx2][ft]) / 2
                        less = pd.DataFrame()
                        greater = pd.DataFrame()
                        for index, row in xData.iterrows():
                            if (row[ft] > threshold):
                                greater = greater.append(row)
                            else:
                                less = less.append(row)
                        # print("\nGreater, less", len(greater), len(less))

                        e = self.gainRatio(xData, yData, [less, greater])

                        if e >= maxEntropy:
                            splitted = [less, greater]
                            maxEntropy = e
                            bestFt = ft
                            bestThreshold = threshold

        return (bestFt, bestThreshold, splitted)

    def gainRatio(self, xData, yData, subsets):
        # def gain(self, unionSet, subsets):
        # input : data and disjoint subsets of it
        # output : information gain
        S = len(xData)

        # calculate impurity before split
        impurityBeforeSplit = self.entropy(xData, yData)
        # print("Impurity before split", impurityBeforeSplit)

        # calculate impurity after split
        weights = [len(subset) / S for subset in subsets]
        impurityAfterSplit = 0
        for i in range(len(subsets)):
            impurityAfterSplit += weights[i] * self.entropy(subsets[i], yData)

        # print("impurity after split", impurityAfterSplit)

        # calculate total gain ratio
        totalGain = impurityBeforeSplit - impurityAfterSplit
        # print("total gain", totalGain)
        return totalGain

    def entropy(self, xData, yData):

        S = len(xData)
        if S == 0:
            return 0

        classCount = {}
        for n in self.classes:
            classCount[n] = 0
        # num_classes = [0 for i in self.classes]

        for index, row in xData.iterrows():
            clas = yData.loc[index].values[0]
            classCount[clas] += 1

        # print(classCount)

        for key, value in classCount.items():
            classCount[key] = value / S
        # print(classCount)

        #Calculate overall entropy
        ent = 0
        for key, value in classCount.items():
            lg = math.log(value,2) if value else 0
            ent += value * lg

        finalEnt = ent * -1

        return finalEnt



