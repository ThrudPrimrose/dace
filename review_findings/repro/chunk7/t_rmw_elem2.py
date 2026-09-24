import sys, copy
import numpy as np, dace
from drv import *
from dace.transformation.passes.vectorization.lift_map_reduction import LiftMapReductionToReduce
N = dace.symbol('N')

@dace.program
def k(a: dace.float64[N], x: dace.float64[N]):
    x[0] = 0.0
    for i in dace.map[0:N]:
        with dace.tasklet:
            xi << x[i]
            ai << a[i]
            xo >> x[i]
            xo = xi + ai

n = 19
rng = np.random.default_rng(0)
a = rng.random(n); x = rng.random(n)
r = x.copy(); r[0] = 0.0; r += a
s0 = k.to_sdfg(simplify=True)
for st in s0.all_states():
    for e in st.edges(): print("  ", getattr(e.src, 'data', e.src.label), "->", getattr(e.dst, 'data', e.dst.label), str(e.data))
p = copy.deepcopy(s0); p.name = "t_rmw_elem2_lift"
print("lift:", LiftMapReductionToReduce(vectorized=True, rmw_only=True).apply_pass(p, {}))
p.validate()
print("lift only:", end=" "); compare({"x": r}, run(p, {"a": a, "x": x}, N=n))
canon = len(sys.argv) > 1
v = vectorize(None, "t_rmw_elem2_vec%d" % canon, sdfg=copy.deepcopy(s0), canon=canon)
print("full pipeline:", end=" "); compare({"x": r}, run(v, {"a": a, "x": x}, N=n))
