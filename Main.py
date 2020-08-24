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
    #load tree
    dot = read_dot('test2params.dot')
    dot = nx.DiGraph(dot)
    mvdd = MVDD(params.hemo, dot, root='PAMN')
    mvdd.featureDict = params.hemoDict

    #Predict score
    score, path = mvdd.predictScore(paramDict)

    return 'test2params.pdf', score, path #will be displayed on webpage


#Expects a param dict of 119 parameters
def runAllData(paramDict):
    pass




def main():
    pass
    # runHemo()


if __name__ == "__main__":
    main()