import numpy as np, dace
from drv import *
N = dace.symbol('N')

@dace.program
def k(a: dace.int64[N], b: dace.int64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] / b[i]

sref = k.to_sdfg()
n = 19
a = np.arange(n, dtype=np.int64) * 7 + 1
b = np.full(n, 4, dtype=np.int64)
c = np.zeros(n, dtype=np.float64)
print("unvectorized:", end=" "); compare({"c": a / b}, run(sref, {"a": a, "b": b, "c": c}, N=n))
s = vectorize(k, "t_truediv")
for n_, g in s.all_nodes_recursive():
    if isinstance(n_, dace.nodes.Tasklet): print(g.label, n_.label, "|", n_.code.as_string)
print("vectorized:", end=" "); compare({"c": a / b}, run(s, {"a": a, "b": b, "c": c}, N=n))
