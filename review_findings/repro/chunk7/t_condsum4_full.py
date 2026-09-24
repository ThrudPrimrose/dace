import numpy as np, dace, copy
from drv import *
N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def k(a: dace.float64[N, M], b: dace.float64[N], c: dace.float64[N], s: dace.float64[N]):
    for i in range(N):
        if b[i] > 0.2:
            c[i] = 1.0
        for j in dace.map[0:M]:
            if a[i, j] > 0.5:
                s[i] += a[i, j]

n, m = 5, 13
rng = np.random.default_rng(0)
a = rng.random((n, m)); b = rng.random(n)
c = np.zeros(n)
s = np.zeros(n)
ref = {"s": np.where(a > 0.5, a, 0).sum(axis=1), "c": np.where(b > 0.2, 1.0, 0.0)}
sdfg = k.to_sdfg(simplify=True)
from dace.transformation.passes.canonicalize import canonicalize
c0 = copy.deepcopy(sdfg); c0.name = "t_condsum4_canon"
canonicalize(c0, validate=True)
for n_, g in c0.all_nodes_recursive():
    if g is not None and isinstance(g, dace.SDFGState) and g.sdfg.parent_nsdfg_node is not None:
        print("  inner:", g.label, type(n_).__name__, getattr(n_, 'data', getattr(n_, 'label', '')), [str(e.data) for e in g.out_edges(n_)])
print("canon only:", end=" "); compare(ref, run(c0, {"a": a, "b": b, "c": c, "s": s}, N=n, M=m))
v = vectorize(None, "t_condsum4_vec", sdfg=sdfg)
print("full pipeline:", end=" "); compare(ref, run(v, {"a": a, "b": b, "c": c, "s": s}, N=n, M=m))
