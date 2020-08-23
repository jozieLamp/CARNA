from dd import mdd as _mdd
try:
    import dd.cudd as _bdd
except ImportError:
    import dd.autoref as _bdd

def main():
    bdd = _bdd.BDD() #bdd manager
    # bdd.configure(reordering=True)
    bdd.declare('x', 'y', 'z', 't')

    t = '(x & y) => (! z | ! (y => x))'
    s = 'x & y & (z | t)'
    bdd.add_expr(s)
    # bdd.add_expr(t)

    bdd.dump('test.pdf')






if __name__ == '__main__':
    main()