import numpy as np, dace
from drv import *
N = dace.symbol('N')

@dace.program
def k(a: dace.int64[N], c: dace.int64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] * 2 + 1

s0 = k.to_sdfg(simplify=True)
for n_, g in s0.all_nodes_recursive():
    if isinstance(n_, dace.nodes.Tasklet): print("orig", n_.label, "|", n_.code.as_string)
s = vectorize(k, "t_intmul")
for n_, g in s.all_nodes_recursive():
    if isinstance(n_, dace.nodes.Tasklet): print(g.label, n_.label, "|", n_.code.as_string)
n = 19
a = np.arange(n, dtype=np.int64) + (1 << 60) + 1
c = np.zeros(n, dtype=np.int64)
print("vectorized:", end=" "); compare({"c": a * 2 + 1}, run(s, {"a": a, "c": c}, N=n))
