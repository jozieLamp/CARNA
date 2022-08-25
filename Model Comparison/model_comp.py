#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import warnings
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle, zip_longest, chain
import pickle
from MVDD.MVDD import MVDD
import MVDD.MVDD_Generator as mvGen



def trainCrossVal(xData, yData, model, numFolds, modelName, showIndividualROC=True):
    # make stratified k fold object
    kFold = StratifiedKFold(n_splits=numFolds)

    fprList = []
    tprList = []
    rocList = []

    TPRList = []
    TNRList = []
    PPVList = []
    NPVList = []
    FPRList = []
    FNRList = []
    FDRList = []
    ACCList = []
    AUCList = []

    count = 1
    for train_index, test_index in kFold.split(xData, yData):
        X_train, X_test = xData.iloc[train_index], xData.iloc[test_index]
        y_train, y_test = yData.iloc[train_index], yData.iloc[test_index]

        # fit initial decision tree model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Get Accuracy + Confusion Matrix metrics
        cm = confusion_matrix(y_test ,y_pred)

        # check for missing scores
        if len(set(y_test)) != 5 and len(cm[0]) != 5:
            missing = np.setdiff1d([1, 2, 3, 4, 5], list(set(y_test)))
            missing = missing[0] - 1

            cmList = cm.tolist()
            for row in cmList:
                row.insert(missing, 0)

            cm = np.array(cmList)
            cm = np.insert(cm, missing, np.array([0, 0, 0, 0, 0]), 0)

        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        # Sensitivity, hit rate, recall, or true positive rate
        TPRList.append(TP / (TP + FN))
        # Specificity or true negative rate
        TNRList.append(TN / (TN + FP))
        # Precision or positive predictive value
        PPVList.append(TP / (TP + FP))
        # Negative predictive value
        NPVList.append(TN / (TN + FN))
        # Fall out or false positive rate
        FPRList.append(FP / (FP + TN))
        # False negative rate
        FNRList.append(FN / (TP + FN))
        # False discovery rate
        FDRList.append(FP / (TP + FP))
        # Overall accuracy
        mvddAcc = (TP + TN) / (TP + FP + FN + TN)
        ACCList.append(mvddAcc)

        # calculate average roc across all classes
        y_score = label_binarize(y_pred, classes=[1 ,2 ,3 ,4 ,5])
        y_test = label_binarize(y_test, classes=[1 ,2 ,3 ,4 ,5])
        fpr, tpr, roc_auc = getClassROC(y_test ,y_score)
        fprList.append(fpr)
        tprList.append(tpr)
        rocList.append(roc_auc)

        AUCList.append(list(roc_auc.values()))

        # show individual fold roc curves
        if showIndividualROC:
            getIndividualROCGraph(y_test, y_score, count, modelName)

        count += 1

    aveFPR = getDictionaryAverages(fprList)
    aveTPR = getDictionaryAverages(tprList)
    ave_roc_auc = getDictionaryAverages(rocList, hasList=False)

    #     getAverageROCGraph(aveFPR, aveTPR, ave_roc_auc, modelName)

    # Get full metrics
    n = len(xData)

    metricLst = [modelName, np.nanmean(TPRList), np.nanstd(TPRList) * 2, CI(np.nanmean(TPRList), n),
                  np.nanmean(TNRList), np.nanstd(TNRList) * 2, CI(np.nanmean(TNRList), n),
                  np.nanmean(PPVList), np.nanstd(PPVList) * 2, CI(np.nanmean(PPVList), n),
                  np.nanmean(NPVList), np.nanstd(NPVList) * 2, CI(np.nanmean(NPVList), n),
                  np.nanmean(FPRList), np.nanstd(FPRList) * 2, CI(np.nanmean(FPRList), n),
                  np.nanmean(FNRList), np.nanstd(FNRList) * 2, CI(np.nanmean(FNRList), n),
                  np.nanmean(ACCList), np.nanstd(ACCList) * 2, CI(np.nanmean(ACCList), n),
                  np.nanmean(AUCList), np.nanstd(AUCList) * 2, CI(np.nanmean(AUCList), n) ]

    return model, metricLst


# Validation Method
def performValidation(model, data, labels, modelName):
    y_pred = model.predict(data)
    # print("y_pred", y_pred)
    y_test = labels

    FP = confusion_matrix(y_test ,y_pred).sum(axis=0) - np.diag(confusion_matrix(y_test ,y_pred))
    FN = confusion_matrix(y_test ,y_pred).sum(axis=1) - np.diag(confusion_matrix(y_test ,y_pred))
    TP = np.diag(confusion_matrix(y_test ,y_pred))
    TN = confusion_matrix(y_test ,y_pred).sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP /(TP +FN)
    TPR = TPR[~np.isnan(TPR)]
    # Specificity or true negative rate
    TNR = TN /(TN +FP)
    TNR = TNR[~np.isnan(TNR)]
    # Precision or positive predictive value
    PPV = TP /(TP +FP)
    PPV = PPV[~np.isnan(PPV)]
    # Negative predictive value
    NPV = TN /(TN +FN)
    NPV = NPV[~np.isnan(NPV)]
    # Fall out or false positive rate
    FPR = FP /(FP +TN)
    FPR = FPR[~np.isnan(FPR)]
    # False negative rate
    FNR = FN /(TP +FN)
    FNR = FNR[~np.isnan(FNR)]
    # False discovery rate
    FDR = FP /(TP +FP)
    FDR = FDR[~np.isnan(FDR)]

    # Overall accuracy
    ACC = (TP +TN ) /(TP +FP +FN +TN)

    # ROC AUC Score
    y_score = label_binarize(y_pred, classes=[1 ,2 ,3 ,4 ,5])
    y_test = label_binarize(labels, classes=[1 ,2 ,3 ,4 ,5])
    fpr, tpr, roc_auc = mvGen.getClassROC(y_test ,y_score)

    aucVal = np.array(list(roc_auc.values()))
    aucVal = aucVal[~np.isnan(aucVal)]

    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # # Plot and save averaged AUC graph
    # plt.figure(figsize=(10, 8))
    # plt.rcParams.update({'font.size': 18})
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='Averaged AUC: {0:0.3f}'
    #                ''.format(roc_auc["micro"]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('1 - Specificity (False Positive Rate)')
    # plt.ylabel('Sensitivity (True Positive Rate)')
    # plt.legend(loc="lower right")
    # plt.savefig("Graphs/ "+ aucName + "Single_Averaged_AUC.png")


    # print("Sensitivity (TPR): %0.3f(±%0.3f)" % (np.mean(TPR), np.std(TPR) * 2))
    # print("Specificity (TNR): %0.3f(±%0.3f)" % (np.mean(TNR), np.std(TNR) * 2))
    # print("Precision (PPV): %0.3f(±%0.3f)" % (np.mean(PPV), np.std(PPV) * 2))
    # print("Negative Predictive Value (NPV): %0.3f(±%0.3f)" % (np.mean(NPV), np.std(NPV) * 2))
    # print("FPR: %0.3f(±%0.3f)" % (np.mean(FPR), np.std(FPR) * 2))
    # print("FNR: %0.3f(±%0.3f)" % (np.mean(FNR), np.std(FNR) * 2))
    # print("Accuracy: %0.3f(±%0.3f)" % (np.mean(ACC), np.std(ACC) * 2))
    # print("Averaged AUC: %0.3f(±%0.3f)" % (np.mean(aucVal), np.std(aucVal) * 2))
    # # print("Micro AUC: %0.3f" % (roc_auc["micro"]))

    n = len(data)

    metricLst = [modelName, np.nanmean(TPR), np.nanstd(TPR) * 2, CI(np.nanmean(TPR), n),
                 np.nanmean(TNR), np.nanstd(TNR) * 2, CI(np.nanmean(TNR), n),
                 np.nanmean(PPV), np.nanstd(PPV) * 2, CI(np.nanmean(PPV), n),
                 np.nanmean(NPV), np.nanstd(NPV) * 2, CI(np.nanmean(NPV), n),
                 np.nanmean(FPR), np.nanstd(FPR) * 2, CI(np.nanmean(FPR), n),
                 np.nanmean(FNR), np.nanstd(FNR) * 2, CI(np.nanmean(FNR), n),
                 np.nanmean(ACC), np.nanstd(ACC) * 2, CI(np.nanmean(ACC), n),
                 np.nanmean(aucVal), np.nanstd(aucVal) * 2, CI(np.nanmean(aucVal), n)]

    return metricLst


# Helper functions
# Return 95% Confidence Interval for Value
def CI(val, n):
    ci = 1.96 * math.sqrt(abs(val - (1 - val)) / n)
    return ci


def getClassROC(y_test, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(5):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 5

    return fpr, tpr, roc_auc


def getIndividualROCGraph(y_test, y_score, foldNum, modelName):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(5)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(5):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 5

    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 18})
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue','palegreen', 'mistyrose'])
    colors = cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='Class {0} AUC = {1:0.2f})'
                       ''.format(i + 1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    # plt.title('ROC Curve for Each Score Classification Fold ' + str(foldNum))
    plt.legend(loc="lower right")

    plt.savefig("Graphs/" + modelName + "ROC for Fold " + str(foldNum) + ".png")


#     plt.show()

def getDictionaryAverages(dictList, hasList=True):
    d = {}
    for k in dictList[0].keys():
        d[k] = tuple(d[k] for d in dictList)

    if hasList:
        finalDict = {}
        # create average
        for key, value in d.items():
            finalDict[key] = list(map(mapAvg, zip_longest(*value)))
    else:
        finalDict = {}

        for key, value in d.items():
            finalDict[key] = np.mean(value)

    return finalDict


# Helper method to get dictinoary averages
def mapAvg(x):
    x = [i for i in x if i is not None]
    return sum(x, 0.0) / len(x)


def getAverageROCGraph(fpr, tpr, roc_auc, modelName):
    plt.figure(figsize=(10, 8))
    # plt.rc('font', size=14)
    plt.rcParams.update({'font.size': 18})
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'palegreen', 'mistyrose'])
    colors = cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='Class {0} AUC = {1:0.2f})'
                       ''.format(i + 1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    # plt.title('Averaged ROC Curve for Each Score Classification')
    plt.legend(loc="lower right")
    plt.savefig("Graphs/" + modelName + "Averaged_ROC.png")
#     plt.show()