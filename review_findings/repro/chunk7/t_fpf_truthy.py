import sys
import numpy as np, dace
from drv import *
N = dace.symbol('N')

@dace.program
def k(b: dace.int64[N], c: dace.int64[N]):
    for i in dace.map[0:N]:
        if b[i]:
            c[i] = 10
        else:
            c[i] = 20

mode = sys.argv[1] if len(sys.argv) > 1 else "fp_factor"
s = vectorize(k, "t_fpf_truthy_" + mode, branch_mode=mode)
for st in s.all_states():
    for n in st.nodes():
        if isinstance(n, dace.nodes.Tasklet) and ("ITE" in n.code.as_string or "cond" in n.label):
            print(n.label, "|", n.code.as_string)
n = 19
b = np.array([0, 3, 1, -2, 0, 5, 0, 1, 7, 0, 0, 2, 1, 1, 0, 9, 4, 0, 1], dtype=np.int64)
c = np.zeros(n, dtype=np.int64)
ref = {"b": b, "c": np.where(b != 0, 10, 20)}
got = run(s, {"b": b, "c": c}, N=n)
compare(ref, got)
