'''
HemoPheno4HF
SCRIPT DESCRIPTION: Main Runner Class for Learning Scores from the Trained MVDDs
CODE DEVELOPED BY: Josephine Lamp, Steven Lamp
ORGANIZATION: University of Virginia, Charlottesville, VA
LAST UPDATED: 8/24/2020
'''

from MVDD.MVDD import MVDD
import MVDD.MVDD_Generator as mvGen
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import *
import Params as params


#Expects param dict of 27 parameters, and select one of 4 outcomes
#Returns a text file location to display the graph, a integer score value and a string phenotype to be displayed
#Outcome can be "ALL", "DEATH" "REHOSPITALIZATION" "READMISSION" (passed in all caps)
def runHemo(paramDict, outcome):

    #get outcome
    if outcome == "READMISSION":
        modelName = 'TreeFiles/Hemo_Readmission'
    elif outcome == "DEATH":
        modelName = 'TreeFiles/Hemo_Death'
    elif outcome == "REHOSPITALIZATION":
        modelName = 'TreeFiles/Hemo_Rehosp'
    else:
        modelName = 'TreeFiles/Hemo_AllOutcomes'

    #load model
    mvdd = mvGen.loadMVDDFromFile(modelName)

    #Predict score
    score, path = mvdd.predictScore(paramDict)

    if score == 5:
        scorePath = "Returned Score of " + str(score) + ", Risk Level: HIGH\nIndicates a >= 40% chance of the outcome " + outcome
    elif score == 4:
        scorePath = "Returned Score of " + str(score) + ", Risk Level: INTERMEDIATE - HIGH\nIndicates a 30-40% chance of the outcome " + outcome
    elif score == 3:
        scorePath = "Returned Score of " + str(score) + ", Risk Level: INTERMEDIATE\nIndicates a 20-30% chance of the outcome " + outcome
    elif score == 2:
        scorePath = "Returned Score of " + str(score) + ", Risk Level: LOW - INTERMEDIATE\nIndicates a 10-20% chance of the outcome " + outcome
    else: #score == 1:
        scorePath = "Returned Score of " + str(score) + ", Risk Level: LOW\nIndicates a < 10% chance of the outcome " + outcome

    imageName = modelName + '.png'

    return imageName, score, path #will be displayed on webpage


#Expects a param dict of 119 parameters and the specified outcome
#Returns a text file location to display the graph, a integer score value and a string phenotype to be displayed
#Outcome can be "ALL", "DEATH" "REHOSPITALIZATION" "READMISSION" (passed in all caps)
def runAllData(paramDict, outcome):

    # get outcome
    if outcome == "READMISSION":
        modelName = 'TreeFiles/AllData_Readmission'
    elif outcome == "DEATH":
        modelName = 'TreeFiles/AllData_Death'
    elif outcome == "REHOSPITALIZATION":
        modelName = 'TreeFiles/AllData_Rehosp'
    else:
        modelName = 'TreeFiles/AllData_AllOutcomes'

    # load model
    mvdd = mvGen.loadMVDDFromFile(modelName)

    # Predict score
    score, path = mvdd.predictScore(paramDict)

    if score == 5:
        scorePath = "Returned Score of " + str(
            score) + ", Risk Level: HIGH\nIndicates a >= 40% chance of the outcome " + outcome
    elif score == 4:
        scorePath = "Returned Score of " + str(
            score) + ", Risk Level: INTERMEDIATE - HIGH\nIndicates a 30-40% chance of the outcome " + outcome
    elif score == 3:
        scorePath = "Returned Score of " + str(
            score) + ", Risk Level: INTERMEDIATE\nIndicates a 20-30% chance of the outcome " + outcome
    elif score == 2:
        scorePath = "Returned Score of " + str(
            score) + ", Risk Level: LOW - INTERMEDIATE\nIndicates a 10-20% chance of the outcome " + outcome
    else:  # score == 1:
        scorePath = "Returned Score of " + str(
            score) + ", Risk Level: LOW\nIndicates a < 10% chance of the outcome " + outcome

    imageName = modelName + '.png'

    return imageName, score, path  # will be displayed on webpage


def main():
    paramDict = {"Age": "60", "BPDIAS": "63", "BPSYS": "80", "CI": "2.02", "CO": "4.52", "CPI": "0.5", "PCWP": "20", "EjF": "20", "HRTRT": "30",
     "MAP": "110", "MIXED": "", "MPAP": "20", "PAD": "10", "PAMN": "", "PAPP": "", "PAS": "11", "PCWPA": "32", "PCWPMN": "90",
     "PCWPMod": "20", "PP": "", "PPP": "", "PPRatio": "0.9", "RAP": "", "RAT": "", "RATHemo": "10", "SVRHemo": "",
     "SVR": ""}

    filename, score, path = runHemo(paramDict, "DEATH")




if __name__ == "__main__":
    main()