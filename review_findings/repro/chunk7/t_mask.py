import sys
import numpy as np, dace
from drv import *
N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def k(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i in range(N):
        a[i][b[i] > 0.5] = 3.0
        a[i, 0] = a[i, 0] + 1.0

rem = sys.argv[1] if len(sys.argv) > 1 else "scalar_postamble"
n, m = 4, 19
rng = np.random.default_rng(0)
a = rng.random((n, m)); b = rng.random((n, m))
r = a.copy(); r[b > 0.5] = 3.0; r[:, 0] += 1
v = vectorize(k, "t_mask_" + rem, remainder=rem)
for n_, g in v.all_nodes_recursive():
    if isinstance(n_, dace.nodes.Tasklet): print(g.label, n_.label, "|", n_.code.as_string.replace("\n", "; "))
print("full:", end=" "); compare({"a": r}, run(v, {"a": a, "b": b}, N=n, M=m))
