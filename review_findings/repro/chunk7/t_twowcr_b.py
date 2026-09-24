import numpy as np, dace, copy
from drv import *
from dace.transformation.passes.canonicalize import canonicalize
N = dace.symbol('N')

@dace.program
def k(a: dace.float64[N], b: dace.float64[N], s: dace.float64[1]):
    for i in dace.map[0:N]:
        s[0] += a[i]
        s[0] += 2.0 * b[i]

n = 19
rng = np.random.default_rng(0)
a = rng.random(n); b = rng.random(n)
ref = {"s": np.array([1.0 + a.sum() + 2 * b.sum()])}
s0 = k.to_sdfg(simplify=True); s0.name = "t_twowcr_plain"
print("plain:", end=" "); compare(ref, run(s0, {"a": a, "b": b, "s": np.ones(1)}, N=n))
s1 = copy.deepcopy(s0); s1.name = "t_twowcr_canon"; canonicalize(s1, validate=True)
print("canon:", end=" "); compare(ref, run(s1, {"a": a, "b": b, "s": np.ones(1)}, N=n))
