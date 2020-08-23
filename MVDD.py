import random
import networkx as nx
from networkx.drawing.nx_pydot import *
import matplotlib as plt
from graphviz import Digraph
import pydot
from itertools import permutations, repeat, combinations_with_replacement

class MVDD:
    def __init__(self, features):
        self.features = features
        self.graph = None

    def generateRandomGraph(self, nodes, maxBranches):
        # dot = Digraph(strict=True)
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

        availableNodes = nodes #nodes available to choose
        childNodes = []

        #start with root
        currNode = random.choice(nodes)  # pick random node
        availableNodes.remove(currNode)
        for nb in range(random.randint(1, maxBranches)): #add edges to other nodes
            selected = random.choice(availableNodes)
            style = random.randint(1, 2)
            if style == 1:
                dot, edgeDict = self.addEdge(dot, currNode, selected, 'solid', edgeDict)
            else:
                dot, edgeDict = self.addEdge(dot, currNode, selected, 'dashed', edgeDict)

            childNodes.append(selected)

        childNodes = list(set(childNodes))

        while childNodes != []:
            dot, childNodes, availableNodes, edgeDict = self.addChildNodes(dot, childNodes, maxBranches, availableNodes, edgeDict)

        return dot

    #add child nodes to dot graph
    def addChildNodes(self, dot, childNodes, maxBranches, availableNodes, edgeDict):
        for c in childNodes:  # remove new parents
            availableNodes.remove(c)

        if availableNodes == []: #no more nodes to add
            dot, edgeDict = self.addTerminalNodes(dot, childNodes, edgeDict)
            return dot, [], availableNodes, edgeDict
        else:
            newChildren = []

            if len(availableNodes) < 6: #can add some terminal nodes
                for currNode in childNodes:

                    for nb in range(random.randint(1, maxBranches)):  # add edges to other nodes
                        if random.randint(1, 2) == 1:
                            selected = random.choice(availableNodes)
                            style = random.randint(1, 2)
                            if style == 1:
                                dot, edgeDict = self.addEdge(dot, currNode, selected, 'solid', edgeDict)
                            else:
                                dot, edgeDict = self.addEdge(dot, currNode, selected, 'dashed', edgeDict)

                            newChildren.append(selected)

                        else:
                            dot, edgeDict = self.addTerminalNodes(dot, childNodes, edgeDict)

            else: #just add internal nodes
                for currNode in childNodes:
                    for nb in range(random.randint(1,maxBranches)): #add edges to other nodes
                        selected = random.choice(availableNodes)
                        style = random.randint(1,2)
                        if style == 1:
                            dot, edgeDict = self.addEdge(dot, currNode, selected, 'solid', edgeDict)
                        else:
                            dot, edgeDict = self.addEdge(dot, currNode, selected, 'dashed', edgeDict)

                        newChildren.append(selected)

            newChildren = list(set(newChildren))
            return dot, newChildren, availableNodes, edgeDict

    # add terminal nodes
    def addTerminalNodes(self, dot, childNodes, edgeDict):
        terms = ["1", "2", "3", "4", "5"]
        for c in childNodes:
            selected = random.choice(terms)
            dot, edgeDict = self.addEdge(dot, c, selected, 'solid', edgeDict)

        return dot, edgeDict

    #Add edges to dot graph
    def addEdge(self, dot, currNode, selected, type, edgeDict):
        key = currNode + selected + type

        if key in edgeDict:
            pass #edge already in graph
        else:
            dot.add_edge(currNode, selected, style=type)
            key = currNode + selected + type
            edgeDict.append(key)

        return dot, edgeDict


    def saveToFile(self, graph, filename='mvdd', format='pdf'):
        dot = to_pydot(graph)

        if format == "png":
            dot.write_png(filename + '.png')
        else:
            dot.write_pdf(filename + '.pdf')