import sys, copy
import numpy as np, dace
from drv import *
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.lift_map_reduction import LiftMapReductionToReduce
from dace.transformation.passes.vectorization.utils.reductions import recognize_map_reduction
N = dace.symbol('N')

@dace.program
def k(a: dace.float64[N], x: dace.float64[N]):
    x[0] = 0.0
    for i in range(N):
        if a[i] > 0.5:
            x[i] = x[i] + a[i]

n = 19
rng = np.random.default_rng(0)
a = rng.random(n); x = rng.random(n)
r = x.copy(); r[0] = 0.0; r += np.where(a > 0.5, a, 0)
s0 = k.to_sdfg(simplify=True)
c = copy.deepcopy(s0); canonicalize(c, validate=True)
for n_, g in c.all_nodes_recursive():
    if isinstance(n_, dace.nodes.MapEntry):
        print("map", n_.map.params, [type(b).__name__ for b in g.all_nodes_between(n_, g.exit_node(n_))], "recog:", recognize_map_reduction(g, n_))
        for e in g.edges(): print("  ", getattr(e.src, 'data', e.src.label), "->", getattr(e.dst, 'data', e.dst.label), str(e.data))
p = copy.deepcopy(c); p.name = "t_rmw_elem3_lift"
print("lift:", LiftMapReductionToReduce(vectorized=True, rmw_only=True).apply_pass(p, {}))
p.validate()
print("lift only:", end=" "); compare({"x": r}, run(p, {"a": a, "x": x}, N=n))
v = vectorize(None, "t_rmw_elem3_vec", sdfg=copy.deepcopy(s0), canon=True)
print("full pipeline:", end=" "); compare({"x": r}, run(v, {"a": a, "x": x}, N=n))
