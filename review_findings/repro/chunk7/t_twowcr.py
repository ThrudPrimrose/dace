import sys
import numpy as np, dace
from drv import *
N = dace.symbol('N')

@dace.program
def k(a: dace.float64[N], b: dace.float64[N], s: dace.float64[1]):
    for i in dace.map[0:N]:
        s[0] += a[i]
        s[0] += 2.0 * b[i]

rem = sys.argv[1] if len(sys.argv) > 1 else "scalar_postamble"
v = vectorize(k, "t_twowcr_nc_" + rem, remainder=rem, canon=False)
n = 19
rng = np.random.default_rng(0)
a = rng.random(n); b = rng.random(n)
ref = {"s": np.array([1.0 + a.sum() + 2 * b.sum()])}
print("full pipeline:", end=" "); compare(ref, run(v, {"a": a, "b": b, "s": np.ones(1)}, N=n))
