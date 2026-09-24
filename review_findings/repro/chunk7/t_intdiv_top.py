import numpy as np, dace
from drv import *
N = dace.symbol('N')

@dace.program
def k(a: dace.int64[N], b: dace.int64[N], c: dace.int64[N], d: dace.int64[1]):
    for i in dace.map[0:N]:
        c[i] = a[i] * 2
    if b[0] != 0:
        d[0] = a[0] % b[0]
    else:
        d[0] = -1

s = vectorize(k, "t_intdiv_top")
for n_, g in s.all_nodes_recursive():
    if isinstance(n_, dace.nodes.Tasklet) and "%" in n_.code.as_string:
        print("state", g.label, "|", n_.code.as_string)
n = 19
a = np.arange(n, dtype=np.int64) * 7 + 1
b = np.zeros(n, dtype=np.int64)
c = np.zeros(n, dtype=np.int64)
d = np.zeros(1, dtype=np.int64)
got = run(s, {"a": a, "b": b, "c": c, "d": d}, N=n)
compare({"c": a * 2, "d": np.array([-1])}, got)
print([ (type(b).__name__, b.label) for b in s.nodes()])
for n_, g in s.all_nodes_recursive():
    if isinstance(n_, dace.nodes.Tasklet): print(g.label, n_.label, "|", n_.code.as_string)
