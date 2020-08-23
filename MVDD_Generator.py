import random
import networkx as nx
from networkx.drawing.nx_pydot import *
import matplotlib as plt
from graphviz import Digraph
import pydot
from itertools import permutations, repeat, combinations_with_replacement
from MVDD import MVDD


#Generate a random graph from a list of nodes, returns a nx graph
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

    availableNodes = nodes #nodes available to choose
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
        dot, childNodes, availableNodes, edgeDict = addChildNodes(dot, childNodes, maxBranches, availableNodes, edgeDict)

    newMvdd = MVDD(features=nodes, dot=dot, root=root)

    return newMvdd

#add child nodes to dot graph
def addChildNodes(dot, childNodes, maxBranches, availableNodes, edgeDict):
    for c in childNodes:  # remove new parents
        availableNodes.remove(c)

    if availableNodes == []: #no more nodes to add
        dot, edgeDict = addTerminalNodes(dot, childNodes, edgeDict)
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
                        dot, edgeDict = addTerminalNodes(dot, childNodes, edgeDict)

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

# add terminal nodes
def addTerminalNodes(dot, childNodes, edgeDict):
    terms = ["1", "2", "3", "4", "5"]
    for c in childNodes:
        selected = random.choice(terms)
        dot, edgeDict = addEdge(dot, c, selected, 'solid', edgeDict, terminal=True)

    return dot, edgeDict

#Add edges to dot graph
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


def traverseGraph(dot):
    for n in nx.bfs_edges(dot, 'PCWPMod'):
        print(n)
        nodeName = n[0]
        print(dot.nodes[nodeName])


    print("edges connected to node", dot.edges('PCWPMod'))
    print(dot.get_edge_data('PP', 'PAPP')) #get label and style of edge

    #UPDATE LABEL LIKE THIS
    dot.edges['PP', 'PAPP', 0]['label'] = 'New Label'
    print("\n\n", dot.edges['PP', 'PAPP', 0])

    # self.saveToFile(dot, "NewTEST")
    # print(dot.get_edge_data('PP', 'PAPP'))  # get label and style of edge



def addGraphParams(mvdd):
    dot = mvdd.dot
    for ed in nx.bfs_edges(dot, mvdd.root):
        print(ed)
        currNode = ed[0]
        lower = mvdd.featureDict[currNode][0]
        upper = mvdd.featureDict[currNode][1]



    print("edges connected to node", dot.edges('PCWPMod'))
    print(dot.get_edge_data('PP', 'PAPP')) #get label and style of edge

    #UPDATE LABEL LIKE THIS
    dot.edges['PP', 'PAPP', 0]['label'] = 'New Label'
    print("\n\n", dot.edges['PP', 'PAPP', 0])

    # self.saveToFile(dot, "NewTEST")
    # print(dot.get_edge_data('PP', 'PAPP'))  # get label and style of edge