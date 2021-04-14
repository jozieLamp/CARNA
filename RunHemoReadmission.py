#Import packages
import pandas as pd
import numpy as np
import warnings
import MVDD.MVDD_Generator as mvGen
warnings.filterwarnings('ignore')

def main():
    #Load original dataframes
    hemoData = pd.read_csv('Data/Preprocessed Data/ESCAPE_Hemo.csv', index_col='ID')
    allScores = hemoData['Score']
    death = hemoData['ScoreDeath']
    rehosp = hemoData['ScoreRehosp']
    readm = hemoData['ScoreReadmission']

    # Preprocess and create training and testing sets
    hemo = hemoData.drop('Score', axis=1)
    hemo = hemo.drop('ScoreDeath', axis=1)
    hemo = hemo.drop('ScoreRehosp', axis=1)
    hemo = hemo.drop('ScoreReadmission', axis=1)
    hemo = hemo.replace(np.inf, 0)
    hemo = hemo.fillna(0)

    xData = hemo

    yData = readm

    modelName = 'Hemo_Readmission'
    mvdd = mvGen.generateTreeCrossValidation(xData=xData, yData=yData, classes=["1", "2", "3", "4", "5"],
                                            learningCriteria='gini', maxLevels=None, minSamplesPerLeaf=1,
                                            modelName=modelName, numFolds=5,
                                            showIndividualROC=True)

    # Get Feature importance
    featureDict = dict(zip(hemo.columns, mvdd.model.feature_importances_))
    featureImp = pd.DataFrame.from_dict(featureDict, orient='index')
    featureImp.rename(columns = {0:'Feature Importance'}, inplace = True)
    featureImp = featureImp.sort_values(by=['Feature Importance'], ascending=False)
    featureImp.to_csv("Graphs/FeatureImportances" + modelName + ".csv")


if __name__ == "__main__":
    main()