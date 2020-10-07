


import pandas as pd
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree

def main():
    # Load original dataframes
    hemoData = pd.read_csv('../Data/Preprocessed Data/Cluster_Hemo.csv', index_col='DEIDNUM')
    allScores = hemoData['Score']
    death = hemoData['ScoreDeath']
    rehosp = hemoData['ScoreRehosp']
    readm = hemoData['ScoreReadmission']

    # Preprocess and create training and testing sets
    hemo = hemoData.drop('Score', axis=1)
    hemo = hemo.drop('ScoreDeath', axis=1)
    hemo = hemo.drop('ScoreRehosp', axis=1)
    hemo = hemo.drop('ScoreReadmission', axis=1)
    hemo = hemo.reset_index()
    hemo = hemo.drop('DEIDNUM', axis=1)

    allScores = allScores.reset_index()
    allScores = allScores.drop('DEIDNUM', axis=1)
    # hemo = hemo.replace(np.inf, 0)
    # hemo = hemo.fillna(0)

    # Make data with missing values
    xTrain, xTest, yTrain, yTest = train_test_split(hemo, allScores, test_size=0.2)
    print(xTrain.shape, yTrain.shape)

    print(xTrain)
    print(yTrain)

    classes = [1, 2, 3, 4, 5]
    dt = DecisionTree(classes)
    dt.fit(xTrain, yTrain)







if __name__ == "__main__":
    main()