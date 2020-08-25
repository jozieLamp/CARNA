'''
HemoPheno4HF
SCRIPT DESCRIPTION: Training process for developing MVDDs
CODE DEVELOPED BY: Josephine Lamp
ORGANIZATION: University of Virginia, Charlottesville, VA
LAST UPDATED: 8/24/2020
'''

from MVDD.MVDD import MVDD
import MVDD.MVDD_Generator as mvGen
import networkx as nx
from networkx.drawing.nx_pydot import *
import Params as params
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import random
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

# Training process to develop random MVDDs
# INPUT = the total number of trees to generate, and how many different parameter sets to generate per new tree added
# OUTPUT = stores developed trees in the "TreeFiles" folder as images and dot files
def generateTrees(numTrees=10, numParamGens=1):
    hemoData = pd.read_csv('Preprocessed Data/Cluster_Hemo.csv')
    realScores = hemoData['Score']

    globalBestAcc = 0

    for t in range(numTrees):
        #generate tree structure
        mvdd = mvGen.generateRandomMVDD(nodes=params.hemo, maxBranches=3)
        mvdd.saveToFile(filename='TreeFiles/tree'+str(t))
        mvdd.saveDotFile(filename='TreeFiles/tree'+str(t))

        bestAcc = None
        for n in range(numParamGens):
            dot = read_dot('TreeFiles/tree'+str(t)+'.dot')
            dot = nx.DiGraph(dot)
            mvParam = MVDD(params.hemo, dot, root=mvdd.root)
            mvParam.featureDict = params.hemoDict

            mvParam = mvGen.addGraphParamsRandom(mvParam, params.clusterHemoScoreDict)

            predScores = []
            for i in range(len(hemoData)):
                row = hemoData.loc[i]
                score, path = mvParam.predictScore(row)
                predScores.append(int(score))

            acc = accuracy_score(realScores, predScores)
            print(acc)

            #save trees if they have better accuracy than previous ones
            if bestAcc == None or acc > bestAcc:
                bestAcc = acc
                mvParam.saveToFile(filename='TreeFiles/treeParams' + str(t), format='pdf')
                mvParam.saveToFile(filename='TreeFiles/treeParams' + str(t), format='png')
                mvParam.saveDotFile(filename='TreeFiles/treeParams' + str(t))

                if acc > globalBestAcc:
                    globalBestAcc = acc

    print("Best Overall Accuracy is", globalBestAcc)

# Training process to develop best MVDDs
# INPUT = the total number of trees to generate, and how many different parameter sets to generate per new tree added
# OUTPUT = stores developed trees in the "TreeFiles" folder as images and dot files
def generateTreeStructures(nodes, numTreesPerPermutation, maxBranches, xData, yData, numRandom):
    paramRanges = params.hemoParamsV1
    relops = params.hemoRelopsV1
    filename = 'TreeFiles/TreeTraining/tree'

    accList = []

    count = 0
    terminalOrder = ["1", "2", "3", "4", "5"]
    perms = list(permutations(terminalOrder, len(terminalOrder)))
    for p in range(len(perms)):
        for t in range(numTreesPerPermutation):
            # generate tree structure
            mvdd = mvGen.generateMVDDFeatureImportance(nodes=nodes, terminalOrder=perms[p], maxBranches=maxBranches)
            mvdd.saveToFile(filename=filename + str(count), format='pdf')
            mvdd.saveToFile(filename=filename + str(count), format='png')
            mvdd.saveDotFile(filename=filename + str(count))

            #Get some sample ranges
            mvParam, usedParams, usedRelops = mvGen.addGraphParams(mvdd, paramRanges, relops, inorder=True)

            #Get accuracy
            predScores = []
            for index, row in xData.iterrows():
                score, path = mvParam.predictScore(row)
                predScores.append(int(score))

            acc = accuracy_score(yData, predScores)
            accList.append([filename + str(count) + '.dot', acc])

            count += 1

        for r in range(numRandom):
            mvdd = mvGen.generateRandomMVDD(nodes=nodes, maxBranches=maxBranches-1)
            mvdd.saveToFile(filename=filename + "Random" + str(count), format='pdf')
            mvdd.saveToFile(filename=filename + "Random" + str(count), format='png')
            mvdd.saveDotFile(filename=filename + "Random" + str(count))

            # Get some sample ranges
            mvParam, usedParams, usedRelops = mvGen.addGraphParams(mvdd, paramRanges, relops, inorder=True)

            # Get accuracy
            predScores = []
            for index, row in xData.iterrows():
                score, path = mvParam.predictScore(row)
                predScores.append(int(score))

            acc = accuracy_score(yData, predScores)
            accList.append([filename + "Random" + str(count) + '.dot', acc])

            count += 1


    accDF = pd.DataFrame(accList, columns=['Filename', 'Accuracy'])
    accDF = accDF.sort_values(by=['Accuracy'], ascending=False)

    return accDF

def runTrees():
    #Load data
    hemoData = pd.read_csv('Preprocessed Data/Cluster_Hemo.csv')
    realScores = hemoData['Score']

    # Preprocess and create training and testing sets
    hemo = hemoData.drop('Score', axis=1)
    hemo = hemo.replace(np.inf, 0)
    hemo = hemo.fillna(0)
    xTrain, xTest, yTrain, yTest = train_test_split(hemo, realScores, test_size=.2)

    accDF = generateTreeStructures(nodes=params.hemoFeatureImportance, numTreesPerPermutation=5, maxBranches=4, xData=xTrain, yData=yTrain, numRandom=5)
    accDF.to_csv('AccuracyDFTrees.csv')

    print(accDF)


# Training process to find best set of parameters for a given tree
# INPUT = the .dot tree file name, root node of tree, the training data (xdata / ydata), a dictionary of parameters to try for each feature
#         and a dictionary of relational operators (>=, >, <, <=) for each parameter
# OUTPUT = returns an accuracy score, a dictionary of used params and a dictionary of used relops
def optimizeParams(treeFilename, rootNode, xData, yData, paramRanges, relops):
    dot = read_dot(treeFilename + '.dot')
    dot = nx.DiGraph(dot)
    mvdd = MVDD(params.hemo, dot, root=rootNode)
    mvdd.featureDict = params.hemoDict

    mvParam, usedParams, usedRelops = mvGen.addGraphParams(mvdd, paramRanges, relops, inorder=True)

    mvParam.saveToFile(treeFilename + "Params")
    mvParam.saveDotFile(treeFilename + "Params")

    predScores = []

    for index, row in xData.iterrows():
        score, path = mvParam.predictScore(row)
        predScores.append(int(score))

    acc = accuracy_score(yData, predScores)

    return acc, usedParams, usedRelops


def findBestParams():
    # Load data
    hemoData = pd.read_csv('Preprocessed Data/Cluster_Hemo.csv')
    realScores = hemoData['Score']

    # Preprocess and create training and testing sets
    hemo = hemoData.drop('Score', axis=1)
    hemo = hemo.replace(np.inf, 0)
    hemo = hemo.fillna(0)
    xTrain, xTest, yTrain, yTest = train_test_split(hemo, realScores, test_size=.2)

    # NOTE- each node can have up to 4 branches, so each param dict needs to send at least 4 params
    paramRanges = params.hemoParamsV1
    relopChoices = params.hemoRelopsV1

    # selectedTree = 'TreeFiles/tree1' #selected tree to try
    # rootNode = 'PCWP'
    rootNodeList = ['PAS', 'PCWP', 'BPSYS', 'BPSYS', 'PAS', 'CI', 'EjF', 'MPAP', 'HRTRT',
                    'PAPP', 'SVR', 'PAD', 'PCWPMN', 'BPDIAS', 'PAS', 'PPRatio', 'PPRatio', 'PCWPA', 'PAPP', 'EjF']
    accuracy = []
    for i in range(len(rootNodeList)):
        selectedTree = 'TreeFiles/tree' + str(i)
        rootNode = rootNodeList[i]

        # Run param optimization
        acc, usedParams, usedRelops = optimizeParams(treeFilename=selectedTree, rootNode=rootNode, xData=xTrain,
                                                     yData=yTrain, paramRanges=paramRanges, relops=relopChoices)
        accuracy.append("tree" + str(i) + ": " + str(acc))
        print("Accuracy is", acc)
        print("selected tree is", selectedTree)
        print("Rootnode is ", rootNode)
        # print(usedParams)
        # print(usedRelops)

    print(accuracy)
    # 0.44


def main():
    runTrees()




if __name__ == "__main__":
    main()