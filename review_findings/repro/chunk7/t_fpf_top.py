import sys
import numpy as np, dace
from drv import *
N = dace.symbol('N')

@dace.program
def k(b: dace.int64[N], c: dace.int64[N], d: dace.int64[1]):
    for i in dace.map[0:N]:
        c[i] = b[i] * 2
    if b[1]:
        d[0] = 10
    else:
        d[0] = 20

mode = sys.argv[1] if len(sys.argv) > 1 else "fp_factor"
s = vectorize(k, "t_fpf_top_" + mode, branch_mode=mode)
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet) and ("ITE" in n.code.as_string):
        print(n.label, "|", n.code.as_string)
n = 19
b = np.array([0, 3, 1, -2, 0, 5, 0, 1, 7, 0, 0, 2, 1, 1, 0, 9, 4, 0, 1], dtype=np.int64)
c = np.zeros(n, dtype=np.int64)
d = np.zeros(1, dtype=np.int64)
ref = {"b": b, "c": b * 2, "d": np.array([10 if b[1] else 20])}
got = run(s, {"b": b, "c": c, "d": d}, N=n)
compare(ref, got)
