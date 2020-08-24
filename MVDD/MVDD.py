'''
HemoPheno4HF
SCRIPT DESCRIPTION: Multi Valued Decision Diagram Object Class
CODE DEVELOPED BY: Josephine Lamp
ORGANIZATION: University of Virginia, Charlottesville, VA
LAST UPDATED: 8/24/2020
'''

import pydot
import networkx as nx
from networkx.drawing.nx_pydot import *
import collections
import math


class MVDD:
    def __init__(self, features, dot, root=None, featureDict={}):
        self.features = features #list of features
        self.dot = dot #network x graph, representing MVDD
        self.root = root #root node of tree
        self.featureDict = featureDict #feature dictionary with ranges for each of the features

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

    # Predicts score from a dictionary of feature values
    # INPUT = feature dictionary of actual data values
    # OUTPUT = predicted score and the final path illustrating the used phenotype
    def predictScore(self, featureDict):
        paths = self.get_all_tree_paths()
        # print(paths)

        scores = []
        truePaths =[]
        for p in paths:
            # print(p)
            truthVal, score = self.evaluateTruthValue(p, featureDict)
            # print(truthVal)
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

    # Evaluates the truth value of a path (whether the path is applicable, i.e. true, to the supplied data or not)
    # INPUT = Tree path and data in feature dict
    # OUTPUT = returns true or false and the final score terminal node
    def evaluateTruthValue(self, path, featureDict):
        orList = []
        for i in range(0,len(path),4):

            if i+1 == len(path): #reached terminal node
                if orList != []:
                    if True in orList:
                        orList = []
                    else:
                        # print("false at final or node")
                        return False, path[-1]

            else:#not terminal node
                value = featureDict[path[i]]
                # print(path[i], " is", value)

                if not math.isnan(value):

                    bound = path[i+1]
                    param = float(path[i+2])
                    op = path[i+3]


                    if op == '&':
                        if orList != []:
                            #finish previous or list
                            if bound == '<=':
                                if value <= param:
                                    orList.append(True)
                                else:
                                    orList.append(False)
                            else:
                                if value >= param:
                                    orList.append(True)
                                else:
                                    orList.append(False)

                            # print(orList)
                            if True in orList:
                                orList = []
                            else:
                                # print("return false at or list")
                                return False, path[-1]

                        if bound == '<=':
                            if not value <= param:
                                return False, path[-1]
                        else:
                            if not value >= param:
                                return False, path[-1]
                    else: #OR op
                        if bound == '<=':
                            if value <= param:
                                orList.append(True)
                            else:
                                orList.append(False)
                        else:
                            if value >= param:
                                orList.append(True)
                            else:
                                orList.append(False)

        return True, path[-1]



    #Get all tree paths from root
    # INPUT = N/A
    # OUTPUT = returns a list of all paths through the mvdd dot tree
    def get_all_tree_paths(self):
        allPaths = []
        for t in ['1', '2', '3', '4', '5']:
            for p in self.all_simple_paths(self.dot, self.root, t):
                path = []

                for i in range(len(p)-1):
                    path.append(p[i])
                    style = self.dot.get_edge_data(p[i], p[i+1], 0)['style']
                    bound = self.dot.get_edge_data(p[i], p[i + 1], 0)['op']
                    param = self.dot.get_edge_data(p[i], p[i + 1], 0)['param']
                    op = '&' if style == 'solid' else '|'

                    path.append(bound)
                    path.append(param)
                    path.append(op)

                path.append(p[len(p)-1])
                allPaths.append(path)

        return allPaths

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