import numpy as np, dace
from drv import *
N = dace.symbol('N')

@dace.program
def k(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] * 2.0
    for j in range(1, N):
        if a[j] > 0.5:
            continue
        b[j] = b[j - 1] + a[j]

s = vectorize(k, "t_continue")
print("vectorized OK")
