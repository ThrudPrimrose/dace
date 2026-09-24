import numpy as np, dace
from drv import *
N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def k(a: dace.float64[N, M], b: dace.float64[N], s: dace.float64[N]):
    for i in range(N):
        tmp = b[i] * 2.0
        for j in dace.map[0:M]:
            if a[i, j] > tmp:
                s[i] += a[i, j]

sdfg = k.to_sdfg(simplify=True)
from dace.transformation.passes.canonicalize import canonicalize
canonicalize(sdfg, validate=True)
for n_, g in sdfg.all_nodes_recursive():
    if isinstance(n_, dace.nodes.MapEntry):
        print("map", n_.map.params, "in", g.sdfg.name, "nested" if g.sdfg.parent_nsdfg_node else "top", [ (str(e.data), e.data.dynamic) for e in g.in_edges(g.exit_node(n_))])
from dace.transformation.passes.vectorization.lift_map_reduction import LiftMapReductionToReduce
print("lift", LiftMapReductionToReduce(vectorized=True, pure_wcr_only=True, nested_only=True, wcr_free_output=True).apply_pass(sdfg, {}))
sdfg.validate()
sdfg.name = "t_condsum3_lift"
n, m = 5, 13
rng = np.random.default_rng(0)
a = rng.random((n, m)); b = rng.random(n) * 0.4
s = np.zeros(n)
ref = np.where(a > 2 * b[:, None], a, 0).sum(axis=1)
print("after lift only:", end=" "); compare({"s": ref}, run(sdfg, {"a": a, "b": b, "s": s}, N=n, M=m))
