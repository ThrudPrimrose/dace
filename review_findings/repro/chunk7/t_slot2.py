import sys, copy
import numpy as np, dace
from drv import *
from dace.transformation.passes.vectorization.reduction_scalar_local_prep import PrepareReductionForWidening
N = dace.symbol('N')

@dace.program
def k(a: dace.float64[N], b: dace.float64[N], s: dace.float64[4]):
    for i in dace.map[0:N]:
        s[3] += a[i]
        s[3] += b[i]

n = 19
rng = np.random.default_rng(0)
a = rng.random(n); b = rng.random(n)
s = np.ones(4)
r = s.copy(); r[3] += a.sum() + b.sum()
s0 = k.to_sdfg(simplify=True)
st = [x for x in s0.all_states() if any(isinstance(m, dace.nodes.MapExit) for m in x.nodes())][0]
mx = [m for m in st.nodes() if isinstance(m, dace.nodes.MapExit)][0]
print("map exit in-edges:", [(e.dst_conn, str(e.data)) for e in st.in_edges(mx)])
p = copy.deepcopy(s0); p.name = "t_slot2_prep"
print("prep:", PrepareReductionForWidening().apply_pass(p, {}))
p.validate()
print("prep only:", end=" "); compare({"s": r}, run(p, {"a": a, "b": b, "s": s}, N=n))
