import MVDD_Generator as mvGen
from MVDD import MVDD
import networkx as nx
from networkx.drawing.nx_pydot import *
import Params as params

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


def josieTest():


    # mvdd = mvGen.generateRandomMVDD(nodes=hemo, maxBranches=3)
    # mvdd.saveToFile(filename='test1')
    # mvdd.saveDotFile('test1')

    # mvdd.featureDict = hemoDict

    dot = read_dot('test.dot')

    mvdd = MVDD(params.hemo, dot, root="PAS")
    mvdd.featureDict = params.hemoDict

    mvGen.addGraphParams(mvdd)

def main():
    josieTest()


if __name__ == "__main__":
    main()