import numpy as np, dace
from drv import *
N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def k(a: dace.float64[N, M], s: dace.float64[N]):
    for i in range(N):
        for j in dace.map[0:M]:
            if a[i, j] > 0.5:
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
sdfg.name = "t_condsum_lift"
n, m = 5, 13
rng = np.random.default_rng(0)
a = rng.random((n, m))
s = np.zeros(n)
ref = np.where(a > 0.5, a, 0).sum(axis=1)
print("after lift only:", end=" "); compare({"s": ref}, run(sdfg, {"a": a, "s": s}, N=n, M=m))
