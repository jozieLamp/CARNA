import MVDD_Generator as mvGen
from MVDD import MVDD
import networkx as nx
from networkx.drawing.nx_pydot import *
import Params as params
import pandas as pd
import copy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def generateTrees(numTrees=2, numParamGens=3):
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
                mvParam.saveToFile(filename='TreeFiles/treeParams' + str(t))
                mvParam.saveDotFile(filename='TreeFiles/treeParams' + str(t))

                if acc > globalBestAcc:
                    globalBestAcc = acc

    print("Best Overall Accuracy is", globalBestAcc)





def main():
    generateTrees()


if __name__ == "__main__":
    main()