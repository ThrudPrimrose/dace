import numpy as np, dace, sys
from drv import *
N = dace.symbol('N'); M = dace.symbol('M')
@dace.program
def f(A: dace.float64[N, M], y: dace.float64[N]):
    for i, j in dace.map[0:N, 0:M]:
        y[i] += A[i, j]
strat = sys.argv[1]; widths = tuple(int(x) for x in sys.argv[2].split(','))
s = f.to_sdfg(simplify=True)
vec(s, widths=widths, remainder_strategy=strat)
print("tile ops", ntile(s))
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.MapEntry): print(n.map.label, n.map.range)
for (n, m) in [(5, 7), (4, 8), (3, 2)]:
    A = np.random.rand(n, m); y = np.ones(n)
    s(A=A, y=y, N=n, M=m)
    print(n, m, "y", np.allclose(y, 1 + A.sum(-1)))
