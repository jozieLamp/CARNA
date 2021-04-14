#Import packages
import pandas as pd
import numpy as np
import warnings
import MVDD.MVDD_Generator as mvGen
warnings.filterwarnings('ignore')

def main():
    # Load original dataframes
    allData = pd.read_csv('Data/Preprocessed Data/ESCAPE_AllData.csv', index_col='ID')
    allScores = allData['Score']
    death = allData['ScoreDeath']
    rehosp = allData['ScoreRehosp']
    readm = allData['ScoreReadmission']

    # Preprocess and create training and testing sets
    alld = allData.drop('Score', axis=1)
    alld = alld.drop('ScoreDeath', axis=1)
    alld = alld.drop('ScoreRehosp', axis=1)
    alld = alld.drop('ScoreReadmission', axis=1)
    alld = alld.replace(np.inf, 0)
    alld = alld.fillna(0)

    xData = alld

    yData = allScores

    modelName = 'AllData_AllOutcomes'
    mvdd = mvGen.generateTreeCrossValidation(xData=xData, yData=yData, classes=["1", "2", "3", "4", "5"],
                                             learningCriteria='gini', maxLevels=None, minSamplesPerLeaf=1,
                                             modelName=modelName, numFolds=5,
                                             showIndividualROC=True)

    # Get Feature importance
    featureDict = dict(zip(alld.columns, mvdd.model.feature_importances_))
    featureImp = pd.DataFrame.from_dict(featureDict, orient='index')
    featureImp.rename(columns={0: 'Feature Importance'}, inplace=True)
    featureImp = featureImp.sort_values(by=['Feature Importance'], ascending=False)
    featureImp.to_csv("Graphs/FeatureImportances" + modelName + ".csv")


if __name__ == "__main__":
    main()