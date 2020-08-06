from dd import mdd as _mdd
# from dd.bdd import BDD
from dd.autoref import BDD

def main():
    # dvars = dict(
    #     x=dict(level=0, len=200),
    #     y=dict(level=1, len=10))
    # mdd = _mdd.MDD(dvars)

    bits = dict(x=2, y0=20, y1=10)
    bdd = BDD(bits)
    u = bdd.add_expr('x \/ (~ y0 /\ y1)')
    bdd.incref(u)

    # convert BDD to MDD
    ints = dict(
        x=dict(level=1, len=2, bitnames=['x']),
        y=dict(level=0, len=25, bitnames=['y0', 'y1']))
    mdd, umap = _mdd.bdd_to_mdd(bdd, ints)

    # map node `u` from BDD to MDD
    v = umap[abs(u)]

    s = mdd.to_expr(v)
    print(s)

    pd = _mdd.to_pydot(mdd)




if __name__ == '__main__':
    main()