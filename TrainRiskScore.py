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

# Training process to develop best MVDDs
# INPUT = the total number of trees to generate, and how many different parameter sets to generate per new tree added
# OUTPUT = stores developed trees in the "TreeFiles" folder as images and dot files
def generateTrees(numTrees=10, numParamGens=5):
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

            mvParam = mvGen.addGraphParams(mvParam, params.clusterHemoScoreDict)

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
                mvParam.saveDotFile(filename='TreeFiles/treeParams' + str(t))

                if acc > globalBestAcc:
                    globalBestAcc = acc

    print("Best Overall Accuracy is", globalBestAcc)


def test():
    hemoData = pd.read_csv('Preprocessed Data/Cluster_Hemo.csv')
    realScores = hemoData['Score']

    dot = read_dot('test2params.dot')
    dot = nx.DiGraph(dot)
    mvParam = MVDD(params.hemo, dot, root='PAMN')
    mvParam.featureDict = params.hemoDict

    row = hemoData.loc[0]
    score, path = mvParam.predictScore(row)
    print(score, path)






def main():
    generateTrees()
    # test()

if __name__ == "__main__":
    main()