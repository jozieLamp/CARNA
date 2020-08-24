from graphviz import Digraph


def testGraph():
    dot = Digraph(comment='The Round Table')

    dot.node('A', 'King Arthur')
    dot.node('B', 'Sir Bedevere the Wise')
    dot.node('L', 'Sir Lancelot the Brave')

    dot.edges(['AB', 'AL'])
    dot.edge('B', 'L', constraint='false')

    print(dot.source)

    dot.render('test-round-table.gv', view=True)

def cardGraph():
    dot = Digraph(comment='CardMTDD')

    dot.node('Age')
    dot.node('EjF')
    dot.node('RAP', 'RAP')
    dot.node('RAPBad', 'RAP')
    dot.node('PAD')
    dot.node('PAS')
    dot.node('PCWP')
    dot.node('CI')
    dot.node('BPSYS')
    dot.node('BPSYS2', 'BPSYS')
    dot.node('HR')

    dot.node('Mixed Venous')
    dot.node('Mixed Venous Bad', 'Mixed  Venous')
    dot.node('Pulse Pressure')
    dot.node('MAP')
    dot.node('MPAP')
    dot.node('RAT', 'Ratio MPAP to MAP')

    #Terminal nodes
    dot.node("1", shape="box")
    dot.node("2", shape="box")
    dot.node("3", shape="box")
    dot.node("4", shape="box")
    dot.node("5", shape="box")

    # dot.edges(['AgeEjf', 'AgeRAP'])
    dot.edge('Age', 'Mixed Venous', label="<40")
    dot.edge('Age', 'RAP', label="40-60")
    dot.edge('Age', 'RAPBad', label=">60")

    dot.edge('EjF', 'RAP', label="55-65")
    dot.edge('EjF', 'RAPBad', label="<55", color='red')

    dot.edge('RAP', 'Mixed Venous', label="<10")
    dot.edge('RAP', 'Mixed Venous Bad', label=">10", color='red', style='dashed')
    dot.edge('Mixed Venous', 'MPAP', label="60-75")
    dot.edge('Mixed Venous', 'BPSYS2', label="<60 or >75", color='red', style='dotted')

    dot.edge('MPAP', 'PAS', label='20-25')
    dot.edge('BPSYS2', 'PAD', label='90-150')
    dot.edge('PAS', 'CI', label='<10')
    dot.edge('PAS', 'PAD', label='>10', color='red')
    dot.edge('CI', 'PCWP', label='>2.2')
    dot.edge('CI', 'RAT', label='<1.2', color='red')
    dot.edge('CI', '2', label='<2.2', color='red')
    dot.edge('RAT', 'MAP', label='<10 or >20', color='red')
    dot.edge('RAT', 'HR', label='10-20')
    dot.edge('HR', 'MAP', label='<60 or >100', color='red')
    dot.edge('CI', 'RAT', label='<2.2', color='red')
    dot.edge('HR', '3', label='<70 or >90', color='red')
    dot.edge('HR', 'Pulse Pressure', label='70-90')

    dot.edge('PCWP', 'Pulse Pressure', label='<=15')
    dot.edge('PCWP', 'HR', label='>15', color='red')
    dot.edge('Pulse Pressure', '1', label='40-60')
    dot.edge('Pulse Pressure', '2', label='<40 or >60', color='red')

    dot.edge('MPAP', 'BPSYS2', label='>25', color='red')
    dot.edge('BPSYS2', '3', label='<90 or >150', color='red')
    dot.edge('MAP', '4', label='20-25')
    dot.edge('MAP', '5', label='>25', color='red')

    dot.edge('PAD', '3', label='>15',color='red')
    dot.edge('PAD', '2', label='<=15')

    dot.edge('RAPBad', 'Mixed Venous Bad', label=">10", color='red')
    dot.edge('RAPBad', 'BPSYS2', label="<10")
    dot.edge('Mixed Venous Bad', 'BPSYS', label="<60 or >75", color='red')
    dot.edge('BPSYS', '5', label="<90 or >150", color='red')
    dot.edge('BPSYS', 'MAP', label="90-150")


    print(dot.source)

    dot.render('CardMTDD.gv', view=True)


def main():
    cardGraph()


if __name__ == "__main__":
    main()



