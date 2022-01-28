'''
HemoPheno4HF
SCRIPT DESCRIPTION: Multi Valued Decision Diagram Object Class
CODE DEVELOPED BY: Josephine Lamp
ORGANIZATION: University of Virginia, Charlottesville, VA
LAST UPDATED: 10/14/2020
'''

import pydot
import networkx as nx
from networkx.drawing.nx_pydot import *
import pandas as pd
import collections
import math
import copy
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class MVDD:
    def __init__(self, features, dot, root=None, model=None, featureDict={}):
        self.features = features #list of features
        self.dot = dot #network x graph, representing MVDD
        self.root = root #root node of tree
        self.featureDict = featureDict #feature dictionary with ranges for each of the features
        self.model = model
        self.terminalIndices = None

    # Save graph to file in specific format
    # INPUT = filename and format
    # OUTPUT = saved graph file in specified format
    def saveToFile(self, filename='mvdd', format='pdf'):
        dt = to_pydot(self.dot)

        if format == "png":
            dt.write_png(filename + '.png')
        else:
            dt.write_pdf(filename + '.pdf')

    # Save dot file for MVDD graph
    # INPUT = filename
    # OUTPUT = .dot file of networkx graph
    def saveDotFile(self, filename='mvdd'):
        dt = to_pydot(self.dot)
        dt.write_dot(filename + ".dot")

    # Return dictionary of node and number edges
    # INPUT = if want to return terminal nodes
    # OUTPUT = dictionary of node and number edges
    def getNumberBranchesPerNode(self, returnTerminals=False):
        dct = {}
        for n in nx.nodes(self.dot):
            if returnTerminals:
                dct[n] = len(self.dot.edges(n))
            else:
                if n not in ['1', '2', '3', '4', '5']:
                    dct[n] = len(self.dot.edges(n))

        return dct

    # Predicts score from a single dictionary of feature values
    # INPUT = feature dictionary of actual data values
    # OUTPUT = predicted score and the final path illustrating the used phenotype
    def predictScore(self, paramDict, returnPath=True):

        #Comparison to model only prediction
        dtScore = self.getModelPrediction(paramDict)
        # print("Decision Tree score:", dtScore)

        #Get all paths through graph
        allPaths = self.getAllPaths(self.terminalIndices)

        #Predict score and get decision path
        predScore, path = self.getDecisionPath(allPaths, paramDict)
        # print("\nFinal Score:", predScore)
        # print("Final Path:", path)

        if returnPath:
            return predScore, path
        else:
            return predScore

    # Predicts a set of scores from a dataframe of values for testing purposes
    # INPUT = dataframe of x values
    # OUTPUT = numpy nd array of predicted scores
    def predictScoreSet(self, xData):
        predScores = []
        for index, row in xData.iterrows():
            dictVals = row.to_dict()
            ps = int(self.predictScore(dictVals, returnPath=False))
            predScores.append(ps)

        return np.asarray(predScores)

    # Predicts a set of scores from a dataframe of values for testing purposes using Decision Tree (NOT MVDD)
    # INPUT = dataframe of x values
    # OUTPUT = numpy nd array of predicted scores
    def predictDTScoreSet(self, xData):
        predScores = []
        for index, row in xData.iterrows():
            dictVals = row.to_dict()
            ps = int(self.getModelPrediction(dictVals))
            predScores.append(ps)

        return np.asarray(predScores)

    # Get prediction from actual decision tree model for comparison (not MVDD)
    # INPUT = feature dictionary
    # OUTPUT = predicted score
    def getModelPrediction(self, paramDict):
        # Convert input into dataframe
        data = {}
        for p in paramDict:
            if paramDict[p] == "":
                data[p] = 0
            else:
                data[p] = float(paramDict[p])

        input = pd.Series(data)
        input = input.to_frame()
        xData = input.T

        predScore = self.model.predict(xData)[0]
        return predScore

    # Get decision path used to determine score
    # INPUT = list of all paths through graph and feature dictionary
    # OUTPUT = predicted score and final path
    def getDecisionPath(self, allPaths, ftDict):
        truePaths = []
        scores = []
        for path in allPaths: #get all paths that evaluate to true on this data
            truthVal, s = self.evaluatePath(path, ftDict)
            if truthVal:
                truePaths.append(path)
                scores.append(s)

        #Check for paths with all ANDs and longest num features
        finalPath = []
        for p in truePaths:
            if ('OR' not in p) and (len(p) > len(finalPath)):
                finalPath = p

        #No AND path, now pick longest path remaining
        if finalPath == []:
            for p in truePaths:
                if len(p) > len(finalPath):
                    finalPath = p

        bestScore = finalPath[-1]
        return bestScore, finalPath


    # Evaluate truth value of path given data
    # INPUT = path and feature dictionary
    # OUTPUT = truth value - true or false and score of path
    def evaluatePath(self, path, ftDict):
        # print(path)

        boolPath = []
        for i in range(0, len(path) - 1, 4):  # iterate through each ft of path
            var = path[i]
            relop = path[i + 1]
            param = path[i + 2]
            op = path[i + 3]

            boolPath.append(self.getTruthValue(ftDict[var], relop, param))
            if op != '==>':
                boolPath.append(op)
            else:
                score = path[i + 4]

        # print("boolist", boolPath)

        truthVal = boolPath[0]
        for b in range(1, len(boolPath), 2):
            op = boolPath[b]
            tr = boolPath[b + 1]
            # print(truthVal, op, tr)

            if op == 'AND':
                truthVal = truthVal and tr
            else:
                truthVal = truthVal or tr

        # print(truthVal)
        return truthVal, score


    # Get truth value from feature, relational operator and parameter
    # INPUT = feature, relational operator and parameter
    # OUTPUT = truth value - true or false
    def getTruthValue(self, ft, relop, param):

        param = float(param)

        if ft == "": #empty variable
            return False
        else:
            value = float(ft)

        if relop == '<=':
            return value <= param
        elif relop == '>=':
            return value >= param
        elif relop == '>':
            return value > param
        else:
            return value < param

    # Get all paths through graph
    # INPUT = terminal indices
    # OUTPUT = list of lists of paths through graph from root to each terminal indice
    def getAllPaths(self, terminalNodes):

        paths = []

        for t in terminalNodes:
            for p in nx.all_simple_edge_paths(self.dot, source=self.root, target=t): #get all paths to that termIndice
                newPath = []
                for edge in range(len(p)): #iterate through edges
                    node0 = p[edge][0]
                    node1 = p[edge][1]

                    if edge + 1 == len(p): #last edge pair

                        lbl = self.dot.nodes[node0]['label'].split('\\n')
                        var0 = lbl[0]

                        lbl2 = self.dot.nodes[node1]['label'].split('\\n')
                        cls = lbl2[-1].replace("class = ", "")
                        cls = cls.replace("\"","")
                        relop, param = self.dot.get_edge_data(node0, node1)['label'].split(" ")
                        op = '==>'

                        newPath.append(var0)
                        newPath.append(relop)
                        newPath.append(param)
                        newPath.append(op)
                        newPath.append(cls)

                    else:
                        lbl = self.dot.nodes[node0]['label'].split('\\n')
                        var0 = lbl[0]
                        relop, param = self.dot.get_edge_data(node0, node1)['label'].split(" ")

                        if self.dot.get_edge_data(node0, node1)['style'] == 'solid':
                            op = 'AND'
                        else:
                            op = 'OR'

                        newPath.append(var0)
                        newPath.append(relop)
                        newPath.append(param)
                        newPath.append(op)

                paths.append(newPath)

        return paths




