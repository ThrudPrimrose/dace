import sys
import numpy as np, dace
from drv import *
N = dace.symbol('N')

@dace.program
def k(a: dace.int64[N], b: dace.int64[N], c: dace.int64[N]):
    for i in dace.map[0:N]:
        if b[i] != 0:
            c[i] = a[i] // b[i]
        else:
            c[i] = -1

s = vectorize(k, "t_intdiv")
n = 19
a = np.arange(n, dtype=np.int64) * 7
b = np.array([0, 3, 1, -2, 0, 5, 0, 1, 7, 0, 0, 2, 1, 1, 0, 9, 4, 0, 1], dtype=np.int64)
c = np.zeros(n, dtype=np.int64)
ref = {"c": np.where(b != 0, a // np.where(b == 0, 1, b), -1)}
got = run(s, {"a": a, "b": b, "c": c}, N=n)
compare(ref, got)
