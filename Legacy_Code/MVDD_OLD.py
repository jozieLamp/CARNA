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
    def predictScore(self, xData):
        predScore = self.model.predict(xData)[0]

        path = self.getDecisionPath(self.terminalIndices, predScore, xData)
        # path = None
        # for p in paths:
        #     truthVal, score = self.evaluatePathValue(p, xData)
        #     if score == predScore:
        #         path = p
        #         break

        # self.getDecisionPath(xData)
        # tree_rules = export_text(self.model, feature_names=list(xData))
        # print(tree_rules)

        return predScore, path


    #Get all tree paths from root
    # INPUT = N/A
    # OUTPUT = returns a list of all paths through the mvdd dot tree
    def getDecisionPath(self, terminalIndices, score, xData):
        allPaths = []

        for t in terminalIndices:
            for p in self.all_simple_paths(self.dot, self.root, t):

                featureList = []

                #get score
                label = self.dot.nodes[p[-1]]['label']
                label = label.replace("\"", "")
                cls = label.split('\\n')[-1]
                cls = cls.replace("class = ", "")

                if str(cls) == str(score):
                    for i in range(len(p)-1):
                        label = self.dot.nodes[p[i]]['label']
                        label = label.replace("\"", "")
                        ft = label.split('\\n')[0]
                        featureList.append(ft)

                    allPaths.append(featureList)

        #get exact path
        # print(allPaths)
        return allPaths[0]






    # def getDecisionPath(self, data):
    #     node_indicator = self.model.decision_path(data)
    #     leaf_id = self.model.apply(data)
    #     n_nodes = self.model.tree_.node_count
    #     children_left = self.model.tree_.children_left
    #     children_right = self.model.tree_.children_right
    #     feature = self.model.tree_.feature
    #     threshold = self.model.tree_.threshold
    #     print(feature, threshold)
    #     leave_id = self.model.apply(data)
    #
    #     sample_id = 0
    #     node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
    #                                         node_indicator.indptr[sample_id + 1]]
    #
    #     print('Rules used to predict sample %s: ' % sample_id)
    #     for node_id in node_index:
    #
    #         if leave_id[sample_id] == node_id:  # <-- changed != to ==
    #             # continue # <-- comment out
    #             print("leaf node {} reached, no decision here".format(leave_id[sample_id]))  # <--
    #
    #         else:  # < -- added else to iterate through decision nodes
    #             if (data[sample_id, feature[node_id]] <= threshold[node_id]):
    #                 threshold_sign = "<="
    #             else:
    #                 threshold_sign = ">"
    #
    #             print("decision id node %s : (X[%s, %s] (= %s) %s %s)"
    #                   % (node_id,
    #                      sample_id,
    #                      feature[node_id],
    #                      data[sample_id, feature[node_id]],  # <-- changed i to sample_id
    #                      threshold_sign,
    #                      threshold[node_id]))



    # Predicts a set of scores from a dataframe of values for testing purposes
    # INPUT = dataframe of x values, y values and flags if want to print confusion matrix or report
    # OUTPUT = accuracy score value
    def predictScoreSet(self, xData, yData, confusionMatrix=False, report=True):
        predScores = self.model.predict(xData)

        if confusionMatrix:
            print(confusion_matrix(yData, predScores))

        if report:
            print(classification_report(yData, predScores))

        return accuracy_score(yData, predScores)



    # Predicts score from a dictionary of feature values
    # INPUT = feature dictionary of actual data values
    # OUTPUT = predicted score and the final path illustrating the used phenotype
    def predictScorePathValue(self, featureDict, terminalIndices):
        paths = self.get_all_tree_paths(terminalIndices)
        # print(paths)

        scores = []
        truePaths =[]
        for p in paths:
            truthVal, score = self.evaluatePathValue(p, featureDict)
            if truthVal:
                scores.append(score)
                truePaths.append(p)

        # print(scores)
        # print(featureDict)

        if scores == []:
            return 5, paths[0]
        else:
            finalScore = max(set(scores), key=scores.count)
            idx = scores.index(finalScore)
            finalPath = truePaths[idx]

            return finalScore, finalPath

    # Evaluate the value of the path
    # INPUT = path and feature dictionary
    # OUTPUT = path and score value
    def evaluatePathValue(self, path, featureDict):
        newPath = []
        path = self.parseOrs(path, featureDict)

        count=0
        while count <= len(path)-3:
            if path[count] == True or path[count] == False or path[count] == '&':
                newPath.append(path[count])
                count += 1
            else:
                ftName = path[count]
                value = featureDict[ftName]
                bound = path[count + 1]
                param = path[count + 2]

                truthVal = self.getTruthValue(value, bound, param)
                newPath.append(truthVal)

                count += 3

        if False in newPath:
            return False, path[-1]
        else:
            return True, path[-1]

    # Helper function for evaluate path val
    # INPUT = path, feature dictionary
    # OUTPUT = path
    def parseOrs(self, path, featureDict):
        newPath = copy.deepcopy(path)
        indices = [i for i, x in enumerate(path) if x == "|"]

        rowIndices = []
        currRow = []
        for i in range(len(indices)):
            if (i+1 != len(indices)) and indices[i] + 4 == indices[i+1]:
                currRow.append(indices[i])
                currRow.append(indices[i+1])
            elif currRow != []:
                currRow = list(sorted(set(currRow)))
                rowIndices.append(currRow)
                currRow = []

        newIndices = list(np.setdiff1d(indices, rowIndices))

        #parse individual ORs in new indices
        for i in newIndices:
            #get truth value of portion before the or operator
            ftName = path[i - 3]
            value = featureDict[ftName]
            bound = path[i-2]
            param = path[i-1]

            truthVal1 = self.getTruthValue(value, bound, param)
            newPath[i-1] = "DELETED"
            newPath[i-3] = "DELETED"
            newPath[i - 2] = "DELETED"

            #get truth value of portion after the or operator
            ftName = path[i + 1]
            value = featureDict[ftName]
            bound = path[i + 2]
            param = path[i + 3]

            truthVal2 = self.getTruthValue(value, bound, param)
            newPath[i + 1] = "DELETED"
            newPath[i + 2] = "DELETED"
            newPath[i + 3] = "DELETED"

            #get final truth value
            if truthVal1 == True or truthVal2 == True:
                newPath[i] = True
            else:
                newPath[i] = False

        #parse group ors
        for group in rowIndices:
            minIdx = group[0]-3
            maxIdx = group[-1]+3

            truthVals = []

            for item in range(minIdx, maxIdx+1,4):
                ftName = path[item]
                value = featureDict[ftName]
                bound = path[item + 1]
                param = path[item+2]

                truthVals.append(self.getTruthValue(value, bound, param))
                newPath[item] = "DELETED"
                newPath[item + 1] = "DELETED"
                newPath[item + 2] = "DELETED"
                if newPath[item +3] == '|':
                    newPath[item + 3] = "DELETED"


            if True in truthVals:
                newPath[maxIdx] = True
            else:
                newPath[maxIdx] = False

        newPath = list(filter(lambda a: a != "DELETED", newPath))
        return newPath

    # Get truth value from feature, relatioan operator and parameter
    # INPUT = feature, relational operator and parameter
    # OUTPUT = truth value - true or false
    def getTruthValue(self, value, bound, param):
        value = float(value)
        param = float(param)
        #fix any bound issues
        bound = bound.replace('\"', '')
        bound = bound.replace('\'', '')
        if bound == '<=':
            return value <= param
        else:
            return value >= param




    # Helper function for getting all tree paths, override of networkx method
    # INPUT = dot graph G, source (starting) node, and target (ending) node, possible cutoff value
    # OUTPUT = returns an iterator over the tree paths
    def all_simple_paths(self,G, source, target, cutoff=None):
        if source not in G:
            raise nx.NodeNotFound('source node %s not in graph' % source)
        if target in G:
            targets = {target}
        else:
            try:
                targets = set(target)
            except TypeError:
                raise nx.NodeNotFound('target node %s not in graph' % target)
        if source in targets:
            return []
        if cutoff is None:
            cutoff = len(G) - 1
        if cutoff < 1:
            return []

        return self._all_simple_paths_multigraph(G, source, targets, cutoff)

    # Second helper function for getting all tree paths, override of networkx method
    # INPUT = dot graph G, source (starting) node, and target (ending) node, possible cutoff value
    # OUTPUT = returns an iterator over the tree paths
    def _all_simple_paths_multigraph(self, G, source, targets, cutoff):
        visited = collections.OrderedDict.fromkeys([source])
        stack = [(v for u, v in G.edges(source))]
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                visited.popitem()
            elif len(visited) < cutoff:
                if child in visited:
                    continue
                if child in targets:
                    yield list(visited) + [child]
                visited[child] = None
                if targets - set(visited.keys()):
                    stack.append((v for u, v in G.edges(child)))
                else:
                    visited.popitem()
            else:  # len(visited) == cutoff:
                for target in targets - set(visited.keys()):
                    count = ([child] + list(children)).count(target)
                    for i in range(count):
                        yield list(visited) + [target]
                stack.pop()
                visited.popitem()

