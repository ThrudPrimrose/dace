import sys, copy
import numpy as np, dace
from drv import *
from dace.transformation.passes.vectorization.reduction_scalar_local_prep import PrepareReductionForWidening
N = dace.symbol('N')

@dace.program
def k(a: dace.float64[N], s: dace.float64[4], c: dace.float64[1]):
    s[3] = 5.0
    for i in dace.map[0:N]:
        s[3] += a[i]
    c[0] = s[3] * 2.0

n = 19
rng = np.random.default_rng(0)
a = rng.random(n)
s = np.ones(4)
r = s.copy(); r[3] = 5.0 + a.sum()
s0 = k.to_sdfg(simplify=True); s0.name = "t_slot3_plain"
print("plain simplified:", end=" "); compare({"s": r, "c": np.array([2 * r[3]])}, run(s0, {"a": a, "s": s, "c": np.zeros(1)}, N=n))
p = copy.deepcopy(s0); p.name = "t_slot3_prep2"
PrepareReductionForWidening().apply_pass(p, {})
for st in p.states():
    print("STATE", st.label)
    for e in st.edges(): print("  ", getattr(e.src, 'data', e.src.label), "->", getattr(e.dst, 'data', e.dst.label), str(e.data))
