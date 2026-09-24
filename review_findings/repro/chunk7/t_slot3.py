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
s0 = k.to_sdfg(simplify=True)
for st in s0.all_states():
    print(st.label, [(type(x).__name__, getattr(x, 'data', x.label)) for x in st.nodes()])
p = copy.deepcopy(s0); p.name = "t_slot3_prep"
print("prep:", PrepareReductionForWidening().apply_pass(p, {}))
p.validate()
for st in p.all_states():
    print(st.label, [(type(x).__name__, getattr(x, 'data', x.label)) for x in st.nodes()])
print("prep only:", end=" "); compare({"s": r, "c": np.array([2 * r[3]])}, run(p, {"a": a, "s": s, "c": np.zeros(1)}, N=n))
v = vectorize(None, "t_slot3_vec", sdfg=copy.deepcopy(s0), canon=False)
print("full (no canon):", end=" "); compare({"s": r, "c": np.array([2 * r[3]])}, run(v, {"a": a, "s": s, "c": np.zeros(1)}, N=n))
v2 = vectorize(None, "t_slot3_vec_canon", sdfg=copy.deepcopy(s0), canon=True)
print("full (canon):", end=" "); compare({"s": r, "c": np.array([2 * r[3]])}, run(v2, {"a": a, "s": s, "c": np.zeros(1)}, N=n))
from dace.transformation.passes.canonicalize import canonicalize
c3 = copy.deepcopy(s0); c3.name = "t_slot3_canon_only"; canonicalize(c3, validate=True)
print("canon only:", end=" "); compare({"s": r, "c": np.array([2 * r[3]])}, run(c3, {"a": a, "s": s, "c": np.zeros(1)}, N=n))
print("prep on canon:", PrepareReductionForWidening().apply_pass(c3, {}))
