'''
HemoPheno4HF
SCRIPT DESCRIPTION: Multi Valued Decision Diagram Object Class
CODE DEVELOPED BY: Josephine Lamp
ORGANIZATION: University of Virginia, Charlottesville, VA
LAST UPDATED: 8/28/2020
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

    # Traverses dot graph in MVDD
    # INPUT = N/A
    # OUTPUT = N/A
    def traverseGraph(self):
        dot = self.dot
        for n in nx.bfs_edges(dot, 'PCWPMod'):
            print(n)
            nodeName = n[0]
            print(dot.nodes[nodeName])

        print("edges connected to node", dot.edges('PCWPMod'))
        print(dot.get_edge_data('PP', 'PAPP'))  # get label and style of edge

        # UPDATE LABEL LIKE THIS
        dot.edges['PP', 'PAPP', 0]['label'] = 'New Label'
        print("\n\n", dot.edges['PP', 'PAPP', 0])

        # self.saveToFile(dot, "NewTEST")
        # print(dot.get_edge_data('PP', 'PAPP'))  # get label and style of edge

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
    def predictScore(self, paramDict):

        #Convert input into dataframe
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
        path = None

        #path = self.getDecisionPath(self.terminalIndices, predScore, xData)

        # path = None
        # for p in paths:
        #     truthVal, score = self.evaluatePathValue(p, xData)
        #     if score == predScore:
        #         path = p
        #         break

        # self.getDecisionPath(xData)
        # tree_rules = export_text(self.model, feature_names=list(xData))
        # print(tree_rules)

        allPaths = self.getAllPaths(self.terminalIndices)

        print(paramDict['PCWPMod'])

        self.getDecisionPath(allPaths, paramDict)

        return predScore, path

    def getDecisionPath(self, allPaths, ftDict):
        truePaths = []
        for path in allPaths:
            truthVal = self.evaluatePath(path, ftDict)
            if truthVal:
                truePaths.append(path)

        print("TRUE PATHS")
        for t in truePaths:
            print(t)

    #return true or false of ftDict on this path
    def evaluatePath(self, path, ftDict):
        print(path)

        boolPath = []
        for i in range(0, len(path) - 1, 4):  # iterate through each ft of path
            var = path[i]
            relop = path[i + 1]
            param = path[i + 2]
            op = path[i + 3]

            boolPath.append(self.getTruthValue(ftDict[var], relop, param))
            if op != '==>':
                boolPath.append(op)

        print("boolist", boolPath)

        truthVal = boolPath[0]
        for b in range(1, len(boolPath), 2):
            op = boolPath[b]
            tr = boolPath[b + 1]
            # print(truthVal, op, tr)

            if op == 'AND':
                truthVal = truthVal and tr
            else:
                truthVal = truthVal or tr

        print(truthVal)
        return truthVal

    #TODO - need to go through and verify that true val is returning properly... especially with missing values

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




