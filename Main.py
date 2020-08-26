'''
HemoPheno4HF
SCRIPT DESCRIPTION: Main Runner Class for Learning Scors from the Trained MVDDs
CODE DEVELOPED BY: Josephine Lamp
ORGANIZATION: University of Virginia, Charlottesville, VA
LAST UPDATED: 8/24/2020
'''

from MVDD.MVDD import MVDD
import MVDD.MVDD_Generator as mvGen
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import *
import Params as params


#Expects param dict of 27 parameters, and select one of 3 outcomes
#Returns a text file location to display the graph, a integer score value and a string phenotype to be displayed
#Outcome can be "all", "Death" "Rehospitalization" "Readmission"
#outcomes passed in all caps
def runHemo(paramDict, outcome):
    modelName = 'MVDD_Test'

    #check for strings in paramDict
    for p in paramDict:
        if paramDict[p] == "":
            paramDict[p] = 0
        else:
            paramDict[p] = float(paramDict[p])

    #Convert input into dataframe
    input = pd.Series(paramDict)
    input = input.to_frame()
    input = input.T

    #load model
    mvdd = mvGen.loadMVDDFromFile(modelName)

    #Predict score
    score, path = mvdd.predictScore(input)
    print(score)
    print(path)

    stringPath = ""
    if path != None:
        path[-2] = '->'
        stringPath = ' '.join(path)
        print(stringPath)

    return 'TreeFiles/treeParams1.png', score, stringPath #will be displayed on webpage


#Expects a param dict of 119 parameters
def runAllData(paramDict):
    pass




def main():
    paramDict = {"Age": "10", "BPDIAS": "63", "BPSYS": "80", "CI": "2.02", "CO": "4.52", "CPI": "0.54", "PCWP": "18", "EjF": "20", "HRTRT": "70",
     "MAP": "", "MIXED": "", "MPAP": "", "PAD": "", "PAMN": "", "PAPP": "", "PAS": "", "PCWPA": "", "PCWPMN": "",
     "PCWPMod": "", "PP": "", "PPP": "", "PPRatio": "", "RAP": "", "RAT": "", "RATHemo": "", "SVRHemo": "",
     "SVR": ""}

    filename, score, path = runHemo(paramDict, "DEATH")



if __name__ == "__main__":
    main()