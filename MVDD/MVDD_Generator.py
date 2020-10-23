'''
HemoPheno4HF
SCRIPT DESCRIPTION: Generation of MVDDs
CODE DEVELOPED BY: Josephine Lamp, Yuxin Wu
ORGANIZATION: University of Virginia, Charlottesville, VA
LAST UPDATED: 10/14/2020
'''

import random
import networkx as nx
from MVDD.MVDD import MVDD
import copy
from collections import OrderedDict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import pickle
import pydotplus
import collections
from networkx.drawing.nx_pydot import *
from MVDD.MVDD import MVDD
import Params as params
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import interp
import numpy as np
import random
from itertools import permutations, combinations
from more_itertools import distinct_permutations
import itertools
import re
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from statistics import mean
from sklearn.model_selection import StratifiedKFold
import numpy.ma as ma
from itertools import zip_longest


'''
MAIN TRAINING CLASS HERE
'''
# MVDD Training and Generation Process using Cross Validation
# INPUT = x and y data, classes predicting, learning splitting criteria, number of max tree levels, minimum number of samples per leaf and name of model to save,
#   number of folds for cross validation and flag if want to show the individual ROC graphs in the cross validation
# OUTPUT = returns a MVDD object class with the created network x dot graph
def generateTreeCrossValidation(xData, yData, classes, learningCriteria='gini', maxLevels=None, minSamplesPerLeaf=5, modelName='MVDD', numFolds=5, showIndividualROC=True):

    #First learn a decision tree classifier to boost the learning process
    dt = DecisionTreeClassifier(criterion=learningCriteria, random_state=100,
                                max_depth=maxLevels, min_samples_leaf=minSamplesPerLeaf)

    #Perform training using cross validation
    dt, mvdd = trainCrossValidation(xData, yData, dt, numFolds, classes, learningCriteria, showIndividualROC, modelName)

    #Save model to file
    pickle.dump(mvdd, open('TreeFiles/' + modelName+'.sav', 'wb'))

    #Save tree to file
    mvdd.saveDotFile('TreeFiles/' +modelName)
    mvdd.saveToFile('TreeFiles/' +modelName, 'pdf')
    mvdd.saveToFile('TreeFiles/' +modelName, 'png')

    return mvdd

# Performs training of MVDDs
# INPUT = x and y data, classes predicting, decision tree model, number of cross validation folds, whether to show individual roc graphs and model name
# OUTPUT = returns trained decision model
def trainCrossValidation(xData, yData, dt, numFolds, classes, learningCriteria, showIndividualROC, modelName):
    #make stratified k fold object
    kFold = StratifiedKFold(n_splits=numFolds)

    bestMVDD = None
    bestAcc = 0

    acc = []
    recall = []
    f1 = []
    precision = []

    fprList = []
    tprList = []
    rocList = []
    count = 1
    for train_index, test_index in kFold.split(xData, yData):
        X_train, X_test = xData.iloc[train_index], xData.iloc[test_index]
        y_train, y_test = yData.iloc[train_index], yData.iloc[test_index]

        #fit initial decision tree model
        dt.fit(X_train, y_train)
        y_pred_orig = dt.predict(X_test)

        #Generate a bunch of MVDDs, get best one
        mvdd = getBestMVDD(dt, X_train, y_train, classes, learningCriteria)

        #Get predictions
        y_pred = mvdd.predictScoreSet(X_test)

        mvddAcc = accuracy_score(y_test,y_pred)
        print("Accuracy DT:", accuracy_score(y_test, y_pred_orig))
        print("Accuracy MVDD:", mvddAcc)

        #Update best MVDD
        if mvddAcc > bestAcc:
            bestAcc = mvddAcc
            bestMVDD = mvdd

        #get accuracy metrics
        acc.append(mvddAcc)
        recall.append(recall_score(y_test,y_pred,average='weighted'))
        f1.append(f1_score(y_test,y_pred,average='weighted'))
        precision.append(precision_score(y_test,y_pred,average='weighted'))

        #calculate average roc across all classes
        y_score = label_binarize(y_pred, classes=[1,2,3,4,5])
        y_test = label_binarize(y_test, classes=[1,2,3,4,5])
        fpr, tpr, roc_auc = getClassROC(y_test,y_score)
        fprList.append(fpr)
        tprList.append(tpr)
        rocList.append(roc_auc)

        #show individual fold roc curves
        if showIndividualROC:
            getIndividualROCGraph(y_test, y_score, count, modelName)

        count += 1

    aveFPR = getDictionaryAverages(fprList)
    aveTPR = getDictionaryAverages(tprList)
    ave_roc_auc = getDictionaryAverages(rocList, hasList=False)

    getAverageROCGraph(aveFPR, aveTPR, ave_roc_auc, modelName)

    print("*****Averaged Final Classification Results*****")
    print("Accuracy: %0.3f(+/- %0.3f)" % (np.mean(acc), np.std(acc) * 2))
    print("Precision: %0.3f(+/- %0.3f)" % (np.mean(precision), np.std(precision) * 2))
    print("Recall: %0.3f(+/- %0.3f)" % (np.mean(recall), np.std(recall) * 2))
    print("F1: %0.3f(+/- %0.3f)" % (np.mean(f1), np.std(f1) * 2))

    return dt, bestMVDD

# Exhaustively generates a bunch of MVDDs from the decision tree model and returns the best one (one with highest accuracy)
# INPUT = decision tree model, x and y data, classes predicting, learning criteria
# OUTPUT = returns best MVDD model
def getBestMVDD(dt, xData, yData, classes, learningCriteria):

    dot_data = tree.export_graphviz(dt,
                                    feature_names=xData.columns,
                                    class_names=classes,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    dot = nx.nx_pydot.from_pydot(graph)
    dot = nx.DiGraph(dot)

    totalEdges = len(dot.edges)

    # Get terminal indices
    terminalIndices = []
    for n in dot.nodes:
        label = dot.nodes[n]['label']
        label = label.replace("\"", "")
        labelSplit = label.split('\\n')[0]
        tokens = labelSplit.split(' ')

        if tokens[0] == learningCriteria:  # NOTE was 'gini'
            terminalIndices.append(n)

    # get count of non-terminal indices
    numTermEdges = 0
    for d in dot.edges:
        if d[0] in terminalIndices or d[1] in terminalIndices:
            numTermEdges += 1
    print("\nTotal edges:", totalEdges, "Nonterminal Edge", totalEdges - numTermEdges)

    totalEdges = totalEdges - numTermEdges

    percentReqdMin= round(0.3 * totalEdges) #number of or edges required: > 10% of number total edges
    percentReqdMax= round(0.6 * totalEdges) #number of or edges required: < 60% of number total edges

    print("Must have more than", percentReqdMin, "and less than", percentReqdMax,"OR Edges")

    mvddList = []
    mvddAcc = []
    edgeList = []

    # get all combos of edges
    edgeOptions = genEdgeCombos(percentReqdMin, percentReqdMax, totalEdges, combinationSize=100000, sampleSize=10000)

    #Exhaustive sample of MVDD edges to try, and get best resulting MVDD
    for edgeOpt in edgeOptions:
        # print(edgeOpt)
        mvdd = convertDecisionTreeToMVDD(dt, xData, classes, learningCriteria, edgeOpt)

        mvddList.append(mvdd)
        y_pred = mvdd.predictScoreSet(xData)
        acc = accuracy_score(yData, y_pred)
        # print("Accuracy Score", acc)
        mvddAcc.append(acc)
        edgeList.append(edgeOpt)

    # get best MVDD
    maxPos = mvddAcc.index(max(mvddAcc))
    mvdd = mvddList[maxPos]
    print("Best edge set:", edgeList[maxPos])

    return mvdd

def genEdgeCombos(minEdges, maxEdges, totalNumEdges, combinationSize, sampleSize):
    edgeCombos = []
    totalPerms = 0

    numEdges = minEdges

    while numEdges < maxEdges:
        lst = []
        for n in range(totalNumEdges):
            if n <= numEdges:
                lst.insert(0, 'dashed')
            else:
                lst.insert(0, 'solid')

        #get all combos of this size list
        count = 0
        perm = distinct_permutations(lst)
        for p in perm:
            edgeCombos.append(list(p))
            count += 1
            totalPerms += 1

            if count > combinationSize:
                break

        numEdges += 1

    if sampleSize == combinationSize:
        return edgeCombos
    else:
        sample = random.sample(edgeCombos, sampleSize)
        return sample


# Convert decision tree to MVDD
# INPUT = decision tree model, xdata, classes, learning criteria used and list of edge choices (for and/or edges)
# OUTPUT = saves a MVDD graph and returns the new MVDD
def convertDecisionTreeToMVDD(dt, xData, classes, learningCriteria, edgeOpt):
    # Convert decision tree into dot graph
    dot_data = tree.export_graphviz(dt,
                                    feature_names=xData.columns,
                                    class_names=classes,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)


    # Recolor the nodes
    colors = ('palegreen', 'honeydew', 'lightyellow', 'mistyrose', 'lightcoral')
    nodes = graph.get_node_list()

    for node in nodes:
        if node.get_name() not in ('node', 'edge'):
            vals = dt.tree_.value[int(node.get_name())][0]
            maxPos = np.argmax(vals)
            node.set_fillcolor(colors[maxPos])

    # Convert decision tree dot data to decision diagram
    dot = nx.nx_pydot.from_pydot(graph)
    dot = nx.DiGraph(dot)

    # Get terminal indices
    terminalIndices = []
    for n in dot.nodes:
        label = dot.nodes[n]['label']
        label = label.replace("\"", "")
        labelSplit = label.split('\\n')[0]
        tokens = labelSplit.split(' ')

        if tokens[0] == learningCriteria:  # NOTE was 'gini'
            terminalIndices.append(n)

    edgeCount = 0

    for n in dot.nodes:
        label = dot.nodes[n]['label']
        label = label.replace("\"", "")
        labelSplit = label.split('\\n')[0]
        tokens = labelSplit.split(' ')
        leftLabel, leftOp, rightLabel, rightOp, param = getLeftRightLabels(tokens)
        # print(leftOp, leftLabel, rightOp, rightLabel)

        if tokens[0] != 'gini':
            nodeLabel = re.sub(labelSplit, '', dot.nodes[n]['label'])
            nodeLabel = nodeLabel.replace("\"", "")
            nodeLabel = tokens[0] + nodeLabel
            dot.nodes[n]['label'] = nodeLabel

        if list(dot.edges(n)) != []:
            leftEdge = list(dot.edges(n))[0]
            rightEdge = list(dot.edges(n))[1]

            #Assign Left Edge
            if leftEdge[0] in terminalIndices or leftEdge[1] in terminalIndices:
                dot.edges[leftEdge[0], leftEdge[1]]['label'] = leftLabel
                dot.edges[leftEdge[0], leftEdge[1]]['op'] = leftOp
                dot.edges[leftEdge[0], leftEdge[1]]['param'] = param
                dot.edges[leftEdge[0], leftEdge[1]]['style'] = 'solid'
                dot.edges[leftEdge[0], leftEdge[1]]['headlabel'] = ""

            else:
                dot.edges[leftEdge[0], leftEdge[1]]['label'] = leftLabel
                dot.edges[leftEdge[0], leftEdge[1]]['op'] = leftOp
                dot.edges[leftEdge[0], leftEdge[1]]['param'] = param
                dot.edges[leftEdge[0], leftEdge[1]]['style'] = edgeOpt[edgeCount]
                dot.edges[leftEdge[0], leftEdge[1]]['headlabel'] = ""
                edgeCount += 1

            #Assign Right Edge
            if rightEdge[0] in terminalIndices or rightEdge[1] in terminalIndices:
                dot.edges[rightEdge[0], rightEdge[1]]['label'] = rightLabel
                dot.edges[rightEdge[0], rightEdge[1]]['op'] = rightOp
                dot.edges[rightEdge[0], rightEdge[1]]['param'] = param
                dot.edges[rightEdge[0], rightEdge[1]]['style'] = 'solid'
                dot.edges[rightEdge[0], rightEdge[1]]['headlabel'] = ""
            else:
                dot.edges[rightEdge[0], rightEdge[1]]['label'] = rightLabel
                dot.edges[rightEdge[0], rightEdge[1]]['op'] = rightOp
                dot.edges[rightEdge[0], rightEdge[1]]['param'] = param
                dot.edges[rightEdge[0], rightEdge[1]]['style'] = edgeOpt[edgeCount]
                dot.edges[rightEdge[0], rightEdge[1]]['headlabel'] = ""
                edgeCount += 1

    # Create MVDD
    mvdd = MVDD(features=xData.columns, dot=dot, root='0', model=dt)
    mvdd.terminalIndices = terminalIndices

    return mvdd

#Helper method to train cross validation
def getDictionaryAverages(dictList, hasList=True):
    d = {}
    for k in dictList[0].keys():
        d[k] = tuple(d[k] for d in dictList)

    if hasList:
        finalDict = {}
        #create average
        for key, value in d.items():
            finalDict[key] = list(map(mapAvg, zip_longest(*value)))
    else:
        finalDict = {}

        for key,value in d.items():
            finalDict[key] = np.mean(value)

    return finalDict

#Helper method to get dictinoary averages
def mapAvg(x):
    x = [i for i in x if i is not None]
    return sum(x, 0.0) / len(x)


# Get the ROC curve for multi classes
# INPUT = x and y data
# OUTPUT = saves an roc graph
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

# Get an averaged ROC curve for multi classes from cross validation
# INPUT = averaged false positive rates, true positvie rates and model name
# OUTPUT = saves an roc graph
def getAverageROCGraph(fpr, tpr, roc_auc, modelName):
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'palegreen', 'mistyrose'])
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC Class {0} (area = {1:0.2f})'
                       ''.format(i + 1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Averaged ROC Curve for Each Score Classification')
    plt.legend(loc="lower right")
    plt.savefig("Graphs/"+ modelName + "Averaged_ROC.png")
    plt.show()

# Get the individual ROC curves for each class
# INPUT = x and y data the fold number and the model name
# OUTPUT = saves an roc graph
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
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','palegreen', 'mistyrose'])
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC Class {0} (area = {1:0.2f})'
                 ''.format(i+1, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Score Classification Fold ' + str(foldNum))
    plt.legend(loc="lower right")

    plt.savefig("Graphs/" + modelName + "ROC for Fold " + str(foldNum) + ".png")
    plt.show()


# Helper methods
def getLeftRightLabels(tokens):
    leftLabel = tokens[1] + " " + tokens[2]
    leftOp = tokens[1]
    param = tokens[2]

    if leftOp == '<=':
        rightOp = '>'
    elif leftOp == '>=':
        rightOp = '<'
    elif leftOp == '>':
        rightOp = '<='
    else:
        rightOp = '>='

    rightLabel = rightOp + " " + param

    return leftLabel, leftOp, rightLabel, rightOp, param

# Load Saved MVDD model from file
# INPUT = model name
# OUTPUT = MVDD data structure
def loadMVDDFromFile(modelName):
    return pickle.load(open(modelName + '.sav', 'rb'))

# Training for finding best set of model params
# INPUT = x and y data and params to try
# OUTPUT = prints best params and their accuracies
def findBestModelParams(xData, yData, params):
    dt = DecisionTreeClassifier()

    gs = GridSearchCV(dt, params)
    gs.fit(xData, yData)

    print("Best parameters set found on training set:")
    print(gs.best_params_)

    y_true, y_pred = yData, gs.predict(xData)
    print(classification_report(y_true, y_pred))

# Generate a random MVDD from a starting list of nodes
# INPUT = list of feature nodes, and the maximum number of branches allowed from each node
# OUTPUT = returns a MVDD object class with the created network x dot graph
def generateRandomMVDD(nodes, maxBranches):

    #Generate dot graph
    dot = nx.DiGraph()
    edgeDict = [] # track edges already added

    # Terminal nodes
    dot.add_node("1", shape="box")
    dot.add_node("2", shape="box")
    dot.add_node("3", shape="box")
    dot.add_node("4", shape="box")
    dot.add_node("5", shape="box")

    #Add nodes to class
    for n in nodes:
        dot.add_node(n)

    availableNodes = copy.deepcopy(nodes) #nodes available to choose
    childNodes = []

    #start with root
    currNode = random.choice(nodes)  # pick random node
    root = currNode
    availableNodes.remove(currNode)
    for nb in range(random.randint(1, maxBranches)): #add edges to other nodes
        selected = random.choice(availableNodes)
        style = random.randint(1, 2)
        if style == 1:
            dot, edgeDict = addEdge(dot, currNode, selected, 'solid', edgeDict)
        else:
            dot, edgeDict = addEdge(dot, currNode, selected, 'dashed', edgeDict)

        childNodes.append(selected)

    childNodes = list(set(childNodes))

    while childNodes != []:
        dot, childNodes, availableNodes, edgeDict = addChildNodesRandom(dot, childNodes, maxBranches, availableNodes, edgeDict)

    newMvdd = MVDD(features=nodes, dot=dot, root=root)

    return newMvdd

# Add child nodes to a dot graph
# INPUT = dot graph, list of child nodes to add, the number of max branches, a list of available nodes to select from, and a dictionary of edges
# OUTPUT = returns the dot graph, the list of new child nodes, the updated list of available nodes and the updated dictionary of edges
def addChildNodesRandom(dot, childNodes, maxBranches, availableNodes, edgeDict):
    for c in childNodes:  # remove new parents
        availableNodes.remove(c)

    if availableNodes == []: #no more nodes to add
        dot, edgeDict = addTerminalNodesRandom(dot, childNodes, edgeDict)
        return dot, [], availableNodes, edgeDict
    else:
        newChildren = []

        if len(availableNodes) < 6: #can add some terminal nodes
            for currNode in childNodes:

                for nb in range(random.randint(2, maxBranches)):  # add edges to other nodes
                    if random.randint(1, 2) == 1:
                        selected = random.choice(availableNodes)
                        style = random.randint(1, 2)
                        if style == 1:
                            dot, edgeDict = addEdge(dot, currNode, selected, 'solid', edgeDict)
                        else:
                            dot, edgeDict = addEdge(dot, currNode, selected, 'dashed', edgeDict)

                        newChildren.append(selected)

                    else:
                        dot, edgeDict = addTerminalNodesRandom(dot, childNodes, edgeDict)

        else: #just add internal nodes
            for currNode in childNodes:
                for nb in range(random.randint(2,maxBranches)): #add edges to other nodes
                    selected = random.choice(availableNodes)
                    style = random.randint(1,2)
                    if style == 1:
                        dot, edgeDict = addEdge(dot, currNode, selected, 'solid', edgeDict)
                    else:
                        dot, edgeDict = addEdge(dot, currNode, selected, 'dashed', edgeDict)

                    newChildren.append(selected)

        newChildren = list(set(newChildren))
        return dot, newChildren, availableNodes, edgeDict

# Add terminal (leaf) nodes to dot graph
# INPUT = dot graph, list of child nodes to add and a dictionary of edges
# OUTPUT = returns the dot graph and the updated dictionary of edges
def addTerminalNodesRandom(dot, childNodes, edgeDict):
    terms = ["1", "2", "3", "4", "5"]
    for c in childNodes:
        selected = random.choice(terms)
        dot, edgeDict = addEdge(dot, c, selected, 'solid', edgeDict, terminal=True)

    return dot, edgeDict

# Generate a random MVDD from a starting list of nodes
# INPUT = list of feature nodes, and the maximum number of branches allowed from each node
# OUTPUT = returns a MVDD object class with the created network x dot graph
def generateMVDDFeatureImportance(nodes, terminalOrder, maxBranches):

    #Generate dot graph
    dot = nx.DiGraph()
    edgeDict = [] # track edges already added

    # Terminal nodes
    dot.add_node("1", shape="box")
    dot.add_node("2", shape="box")
    dot.add_node("3", shape="box")
    dot.add_node("4", shape="box")
    dot.add_node("5", shape="box")

    #Add nodes to class
    for n in nodes:
        dot.add_node(n)

    availableNodes = copy.deepcopy(nodes) #nodes available to choose
    childNodes = []

    #start with root
    currNode = nodes[0]  # pick first node
    root = currNode
    availableNodes.remove(currNode)
    count = 0
    for nb in range(random.randint(2, maxBranches-1)): #add edges to other nodes
        selected = availableNodes[count]
        count += 1
        style = random.randint(1, 2)
        if style == 1:
            dot, edgeDict = addEdge(dot, currNode, selected, 'solid', edgeDict)
        else:
            dot, edgeDict = addEdge(dot, currNode, selected, 'dashed', edgeDict)

        childNodes.append(selected)

    childNodes = list(OrderedDict.fromkeys(childNodes))

    while childNodes != []:
        dot, childNodes, availableNodes, edgeDict = addChildNodes(dot, childNodes, maxBranches, availableNodes, edgeDict, terminalOrder)

    newMvdd = MVDD(features=nodes, dot=dot, root=root)

    return newMvdd

# Add child nodes to a dot graph
# INPUT = dot graph, list of child nodes to add, the number of max branches, a list of available nodes to select from, and a dictionary of edges
# OUTPUT = returns the dot graph, the list of new child nodes, the updated list of available nodes and the updated dictionary of edges
def addChildNodes(dot, childNodes, maxBranches, availableNodes, edgeDict, terminalOrder):
    for c in childNodes:  # remove new parents
        availableNodes.remove(c)

    if availableNodes == []: #no more nodes to add
        dot, edgeDict = addTerminalNodes(dot, childNodes, edgeDict, terminalOrder)
        return dot, [], availableNodes, edgeDict
    else:
        newChildren = []

        if len(availableNodes) < 6: #can add some terminal nodes
            for currNode in childNodes:
                for nb in range(random.randint(2, maxBranches-1)):  # add edges to other nodes
                    if random.randint(1, 2) == 1:
                        firstThird = int(len(availableNodes) / 3)
                        if firstThird < 4:
                            rand = random.randint(0, len(availableNodes)-1)
                        else:
                            rand = random.randint(0, firstThird)
                        selected = availableNodes[rand]
                        style = random.randint(1, 2)
                        if style == 1:
                            dot, edgeDict = addEdge(dot, currNode, selected, 'solid', edgeDict)
                        else:
                            dot, edgeDict = addEdge(dot, currNode, selected, 'dashed', edgeDict)

                        newChildren.append(selected)

                    else:
                        dot, edgeDict = addTerminalNodes(dot, childNodes, edgeDict, terminalOrder)

        else: #just add internal nodes
            for currNode in childNodes:
                for nb in range(random.randint(2,maxBranches-1)): #add edges to other nodes
                    firstThird = int(len(availableNodes) / 3)
                    if firstThird < 4:
                        rand = random.randint(0, len(availableNodes)-1)
                    else:
                        rand = random.randint(0, firstThird)
                    selected = availableNodes[rand]
                    style = random.randint(1,2)
                    if style == 1:
                        dot, edgeDict = addEdge(dot, currNode, selected, 'solid', edgeDict)
                    else:
                        dot, edgeDict = addEdge(dot, currNode, selected, 'dashed', edgeDict)

                    newChildren.append(selected)

        newChildren = list(OrderedDict.fromkeys(newChildren))
        return dot, newChildren, availableNodes, edgeDict

# Add terminal (leaf) nodes to dot graph
# INPUT = dot graph, list of child nodes to add and a dictionary of edges
# OUTPUT = returns the dot graph and the updated dictionary of edges
def addTerminalNodes(dot, childNodes, edgeDict, terminalOrder):
    for c in range(len(childNodes)):
        if c >= len(terminalOrder):
            selected = random.choice(terminalOrder)
        else:
            selected = terminalOrder[c]
        dot, edgeDict = addEdge(dot, childNodes[c], selected, 'solid', edgeDict, terminal=True)

    return dot, edgeDict



# Add an edge to a dot graph
# INPUT = dot graph, current node and selected node to add the edge between, the type of the edge [dashed for or, and solid for and operators], a dictionary of edges,
#         and a check if the edge is connecting a terminal node or not
# OUTPUT = returns the dot graph and the updated dictionary of edges
def addEdge(dot, currNode, selected, type, edgeDict, terminal=False):
    key = currNode + selected + type

    if terminal:
        #check for any other terminal nodes already connected to the node
        termCount = 0
        for i in range(1,6):
            k = currNode + str(i) + type
            if k in edgeDict:
                termCount += 1

        if termCount > 2:
            pass
        else:
            dot.add_edge(currNode, selected, style=type)
            # dot.add_edges_from(currNode, selected, {'style':type}, {'op':'&'}, {'param': '9'})
            key = currNode + selected + type
            edgeDict.append(key)

    else:
        if key in edgeDict:
            pass #edge already in graph
        else:
            dot.add_edge(currNode, selected, style=type)
            key = currNode + selected + type
            edgeDict.append(key)

    return dot, edgeDict

# Updates or adds RANDOM parameters to a dot graph
# INPUT = the MVDD object and a list of average values
# OUTPUT = returns the updated mvdd
def addGraphParamsRandom(mvdd, aveValues):
    dot = mvdd.dot
    # for ed in nx.bfs_edges(dot, mvdd.root):
    for n in nx.nodes(dot):
        # currNode = ed[0]
        # lower = mvdd.featureDict[currNode][0]
        # upper = mvdd.featureDict[currNode][1]

        numEdges = len(dot.edges(n))
        for edg in dot.edges(n):
            if edg[1] in ['1', '2', '3', '4', '5']:
                val = aveValues[n][int(edg[1])-1]
                val = float("{:.2f}".format(val))
                op = random.choice(['<=', '>='])
                label = op + " " + str(val)
                dot.edges[n, edg[1]]['label'] = label
                dot.edges[n, edg[1]]['op'] = op
                dot.edges[n, edg[1]]['param'] = str(val)
            else:
                pos = random.randint(0,4)
                val = aveValues[n][pos]
                val = float("{:.2f}".format(val))
                op = random.choice(["<=", ">="])
                label = op + " " + str(val)
                dot.edges[n, edg[1]]['label'] = label
                dot.edges[n, edg[1]]['op'] = op
                dot.edges[n, edg[1]]['param'] = str(val)

    mvdd.dot = dot

    return mvdd

# Updates or adds specified parameters to a dot graph
# INPUT = the MVDD object, a dictionary with params and the values to add, if the params should be added in order or randomly
# OUTPUT = returns the updated mvdd, the parameters used and the relops used
def addGraphParams(mvdd, paramValues, relopValues, inorder=True):
    usedParams = {}
    usedRelops = {}
    for key in paramValues:
        usedParams[key] = []
        usedRelops[key] = []

    dot = mvdd.dot

    for n in nx.nodes(dot):
        if inorder:
            count = 0
            for edg in dot.edges(n):
                val = paramValues[n][count]
                val = float("{:.2f}".format(val))
                op = relopValues[n][count]
                label = op + " " + str(val)
                dot.edges[n, edg[1]]['label'] = label
                dot.edges[n, edg[1]]['op'] = op
                dot.edges[n, edg[1]]['param'] = str(val)

                usedParams[n].append(val)
                usedRelops[n].append(op)

                count += 1
        else:
            for edg in dot.edges(n):
                idx = random.randint(0, len(relopValues[n])-1)
                val = paramValues[n][idx]
                val = float("{:.2f}".format(val))
                op = relopValues[n][idx]
                label = op + " " + str(val)
                dot.edges[n, edg[1]]['label'] = label
                dot.edges[n, edg[1]]['op'] = op
                dot.edges[n, edg[1]]['param'] = str(val)

                usedParams[n].append(val)
                usedRelops[n].append(op)

    mvdd.dot = dot

    return mvdd, usedParams, usedRelops

# MVDD Training and Generation Process
# INPUT = x and y data, classes predicting, learning splitting criteria, number of max tree levels, minimum number of samples per leaf and name of model to save
# OUTPUT = returns a MVDD object class with the created network x dot graph
def generateTree(xData, yData, classes, learningCriteria='gini', maxLevels=None, minSamplesPerLeaf=5, modelName='MVDD'):

    #First learn a decision tree classifier to boost the learning process
    dt = DecisionTreeClassifier(criterion=learningCriteria, random_state=100,
                                max_depth=maxLevels, min_samples_leaf=minSamplesPerLeaf)
    dt.fit(xData, yData)

    #Convert decision tree into dot graph
    dot_data = tree.export_graphviz(dt,
                                    feature_names=xData.columns,
                                    class_names=classes,
                                    out_file=None,
                                    filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    #Recolor the nodes
    colors = ('palegreen', 'honeydew', 'lightyellow', 'mistyrose', 'lightcoral')
    nodes = graph.get_node_list()

    for node in nodes:
        if node.get_name() not in ('node', 'edge'):
            vals = dt.tree_.value[int(node.get_name())][0]
            maxPos = np.argmax(vals)
            node.set_fillcolor(colors[maxPos])

    # Convert decision tree dot data to decision diagram
    dot = nx.nx_pydot.from_pydot(graph)
    dot = nx.DiGraph(dot)

    # Get terminal indices
    terminalIndices = []
    for n in dot.nodes:
        label = dot.nodes[n]['label']
        label = label.replace("\"", "")
        labelSplit = label.split('\\n')[0]
        tokens = labelSplit.split(' ')

        if tokens[0] == learningCriteria: #NOTE was 'gini'
            terminalIndices.append(n)

    for n in dot.nodes:
        label = dot.nodes[n]['label']
        label = label.replace("\"", "")
        labelSplit = label.split('\\n')[0]
        tokens = labelSplit.split(' ')
        leftLabel, leftOp, rightLabel, rightOp, param = getLeftRightLabels(tokens)

        if tokens[0] != 'gini':
            nodeLabel = re.sub(labelSplit, '', dot.nodes[n]['label'])
            nodeLabel = nodeLabel.replace("\"", "")
            nodeLabel = tokens[0] + nodeLabel
            dot.nodes[n]['label'] = nodeLabel

        for edg in dot.edges(n):
            if edg[0] in terminalIndices or edg[1] in terminalIndices:
                dot.edges[edg[0], edg[1]]['label'] = leftLabel
                dot.edges[edg[0], edg[1]]['op'] = leftOp
                dot.edges[edg[0], edg[1]]['param'] = param
                dot.edges[edg[0], edg[1]]['style'] = 'solid'

                dot.edges[edg[0], edg[1]]['headlabel'] = ""
            else:
                stl = random.choice(['solid', 'dashed'])
                dot.edges[edg[0], edg[1]]['label'] = rightLabel
                dot.edges[edg[0], edg[1]]['op'] = rightOp
                dot.edges[edg[0], edg[1]]['param'] = param
                dot.edges[edg[0], edg[1]]['style'] = stl

                dot.edges[edg[0], edg[1]]['headlabel'] = ""

    #Create MVDD
    mvdd = MVDD(features=xData.columns, dot=dot, root='0', model=dt)
    mvdd.terminalIndices = terminalIndices

    #Save model to file
    pickle.dump(mvdd, open(modelName+'.sav', 'wb'))

    #Save tree to file
    mvdd.saveDotFile(modelName)
    mvdd.saveToFile(modelName, 'pdf')
    mvdd.saveToFile(modelName, 'png')

    return mvdd