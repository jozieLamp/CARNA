import MVDD_Generator as mvGen
from MVDD import MVDD
import networkx as nx
from networkx.drawing.nx_pydot import *
import Params as params
import pandas as pd
import copy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


#Expects param dict of 27 parameters, and select one of 3 outcomes
def runHemo(paramDict, outcome):

    if outcome == 'Death':
        graph = 'test.pdf'
        score = 5
    elif outcome == 'Rehospitalization':
        graph = 'test1.pdf'
        score = 3
    else:
        graph = 'test2.pdf'
        score = 1

    return graph, score #will be displayed on webpage

#Expects a param dict of 119 parameters
def runAllData(paramDict):
    pass


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
                score = mvParam.predictScore(row)
                predScores.append(int(score))

            acc = accuracy_score(realScores, predScores)
            print(acc)

            #save trees if they have better accuracy than previous ones
            if bestAcc == None or acc > bestAcc:
                bestAcc = acc
                mvdd.saveToFile(filename='TreeFiles/treeParams' + str(t))
                mvdd.saveDotFile(filename='TreeFiles/treeParams' + str(t))

                if acc > globalBestAcc:
                    globalBestAcc = acc

    print("Best Overall Accuracy is", globalBestAcc)



def josieTest():

#Main process here

    mvdd = mvGen.generateRandomMVDD(nodes=params.hemo, maxBranches=3)

    mvdd.saveToFile(filename='test2')
    mvdd.saveDotFile(filename='test2')

    # dot = read_dot('test2.dot')
    # dot = nx.DiGraph(dot)
    # mvdd.dot = dot
    mvdd = mvGen.addGraphParams(mvdd, params.clusterHemoScoreDict)
    # mvdd.saveToFile(filename='test2params')
    # mvdd.saveDotFile(filename='test2params')

    # mvdd.saveToFile(filename='test2')
    # mvdd.saveDotFile('test2')

    # mvdd.featureDict = hemoDict



    # dot = read_dot('test2params.dot')
    # dot = nx.DiGraph(dot)
    # print(type(dot))
    # mvdd = MVDD(params.hemo, dot, root="PCWPMod")
    # mvdd.featureDict = params.hemoDict



    #TODO - next need to predict classes from real data
    predScores = []
    hemoData = pd.read_csv('Preprocessed Data/Cluster_Hemo.csv')
    for i in range(len(hemoData)):
        row = hemoData.loc[i]
        score = mvdd.predictScore(row)
        print(score)
        predScores.append(int(score))

    realScores = hemoData['Score']
    print(classification_report(realScores, predScores))

    # print(hemoData.to_dict())





def main():
    generateTrees()


if __name__ == "__main__":
    main()