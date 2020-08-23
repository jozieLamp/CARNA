import pydot
import networkx as nx
from networkx.drawing.nx_pydot import *

class MVDD:
    def __init__(self, features, dot, root=None):
        self.features = features
        self.dot = dot
        self.root = root

    # Save graph to file in specific format
    def saveToFile(self, filename='mvdd', format='pdf'):
        dt = to_pydot(self.dot)

        if format == "png":
            dt.write_png(filename + '.png')
        else:
            dt.write_pdf(filename + '.pdf')

    def saveDotFile(self, filename='mvdd'):
        dt = to_pydot(self.dot)
        dt.write_dot(filename + ".dot")