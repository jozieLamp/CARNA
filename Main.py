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
    score = int(score)

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

    return imageName, score, scorePath, path #will be displayed on webpage


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
    score = int(score)

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

    return imageName, score, scorePath, path  # will be displayed on webpage


def main():

    #Hemo features
    #Age, Gender, Race, EjF, RAP, PAS, PAD, PAMN, PCWP, CO, CI, MIXED, BPSYS, BPDIAS, HRTRT, MAP, MPAP, CPI, PP, PPP, PAPP, SVR, RAT, PPRatio, PAPi, SAPi, CPP, PRAPRat
    paramDict = {'Age': 88.0,'Gender': 2.0,'Race': 1.0,'EjF':25.0,'RAP':24.0,'PAS':42.0,'PAD':24.0,'PAMN':30.0,'PCWP':36.0,'CO':2.2,'CI':1.6,'MIXED':0,'BPSYS':114.0,'BPDIAS':86.0,'HRTRT':105.0,'MAP':171.334,'MPAP':58.0,'CPI':0.6078,'PP':28.0,'PPP': 0.2456,'PAPP':0.4286,'SVR':5357.57,'RAT':0.66,'PPRatio': 0.266,'PAPi': 0.75,'SAPi':0.778,'CPP': 50.0,'PRAPRat':1.167}
    # paramDict = {'Age': 88.0,'Gender': 1.0,'Race': 1.0,'EjF':25.0,'RAP':24.0,'PAS':42.0,'PAD':80.0,'PAMN':30.0,'PCWP':36.0,'CO':2.2,'CI':1.6,'MIXED':0,'BPSYS':100.0,'BPDIAS':86.0,'HRTRT':105.0,'MAP':171.334,'MPAP':58.0,'CPI':0.7078,'PP':28.0,'PPP': 0.2456,'PAPP':0.4286,'SVR':5357.57,'RAT':0.66,'PPRatio': 0.6,'PAPi': 2.0,'SAPi':0.778,'CPP': 50.0,'PRAPRat':1.167}

    filename, score, meaning, path = runHemo(paramDict, "DEATH")

    print("Filename", filename)
    print("Score", score)
    print("Score Meaning", meaning)
    print("Path", path)

    #All data features
    #Age,Gender,Race,Wt,BMI,InitialHospDays,TotalHospDays,NYHA,MLHFS,AF,AlchE,ANGP,ARRH,CARREST,CVD,COPD,DEPR,DIAB,GOUT,HEPT,HTN,MALIG,RENAL,SMOKING,STERD,StrokeTIA,VAHD,VF,VHD,VT,ISCH,NonISCH,CABG,HTRANS,ICD,PACE,PTCI,SixFtWlk,VO2,ALB,ALT,AST,BUN,CRT,DIAL,HEC,HEM,PLA,POT,SOD,TALB,TOTP,WBC,ACE,BET,NIT,DIUR,EjF,BPDIAS,BPSYS,HR,PV,MAP,PP,PPP,PPRatio
    paramDict = {'Age':88.0, 'Gender':2.0, 'Race': 1.0, 'Wt':57.1, 'BMI':26.42417511, 'InitialHospDays':9.0,'TotalHospDays':16.0,'NYHA':4.0,'MLHFS':76.0,
                 'AF':1.0,'AlchE':0.0,'ANGP':1.0,'ARRH':1.0,'CARREST':0.0,'CVD':1.0,'COPD':0.0,'DEPR':0.0,'DIAB':0.0,'GOUT':0.0,'HEPT':0.0,'HTN':0.0,'MALIG':0.0,
                 'RENAL':0.0,'SMOKING':0.0,'STERD':0.0,'StrokeTIA':1.0,'VAHD':1.0,'VF':0.0,'VHD':0.0,'VT':0.0,'ISCH':1.0,'NonISCH':0.0,'CABG':1.0,'HTRANS':1.0,'ICD':0.0,
                 'PACE':1.0,'PTCI':0.0,'SixFtWlk':0.0,'VO2':0.0,'ALB':3.0,'ALT':21.0,'AST':24.0,'BUN':39.0,'CRT':1.5,'DIAL':0.0,'HEC':35.8,'HEM':11.8,'PLA':79.0,
                 'POT':3.5,'SOD':141.0,'TALB':0.7,'TOTP':6.2,'WBC':4.5,'ACE':1.0,'BET':1.0,'NIT':1.0,'DIUR':1.0,'EjF':25.0,'BPDIAS':68.0,'BPSYS':94.0,'HR':104.0,
                 'PV':12.45256301331066,'MAP':139.33333333333334,'PP':26.0,'PPP':0.2765957446808511,'PPRatio':0.25}

    filename, score, meaning, path = runAllData(paramDict, "DEATH")

    print("\nFilename", filename)
    print("Score", score)
    print("Score Meaning:", meaning)
    print("Path", path)

if __name__ == "__main__":
    main()