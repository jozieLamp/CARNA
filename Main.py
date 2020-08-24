'''
HemoPheno4HF
SCRIPT DESCRIPTION: Main Runner Class for Learning Scors from the Trained MVDDs
CODE DEVELOPED BY: Josephine Lamp
ORGANIZATION: University of Virginia, Charlottesville, VA
LAST UPDATED: 8/24/2020
'''

import MVDD_Generator as mvGen
from MVDD import MVDD
import networkx as nx
from networkx.drawing.nx_pydot import *
import Params as params
import pandas as pd
import copy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


#Expects param dict of 27 parameters, and select one of 3 outcomes
#Returns a text file location to display the graph, a integer score value and a string phenotype to be displayed
def runHemo(paramDict, outcome):
    #load tree
    dot = read_dot('test2params.dot')
    dot = nx.DiGraph(dot)
    mvdd = MVDD(params.hemo, dot, root='PAMN')
    mvdd.featureDict = params.hemoDict

    #Predict score
    score, path = mvdd.predictScore(paramDict)
    path[-2] = '->'
    stringPath = ' '.join(path)
    print(stringPath)

    return 'test2params.pdf', score, stringPath #will be displayed on webpage


#Expects a param dict of 119 parameters
def runAllData(paramDict):
    pass




def main():
    pass
    # runHemo()


if __name__ == "__main__":
    main()