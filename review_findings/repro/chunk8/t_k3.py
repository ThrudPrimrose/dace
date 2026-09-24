import numpy as np, dace, sys
from drv import *
N = dace.symbol('N'); M = dace.symbol('M'); L = dace.symbol('L')
@dace.program
def f(A: dace.float64[N, M, L], B: dace.float64[N, M, L], y: dace.float64[N, M]):
    for i, j, k in dace.map[0:N, 1:M, 0:L]:
        B[i, j, k] = A[i, j, k] + A[i, j - 1, k] * 0.5
strat = sys.argv[1]
s = f.to_sdfg(simplify=True)
vec(s, widths=(2, 2, 4), remainder_strategy=strat)
print("tile ops", ntile(s))
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.MapEntry): print(n.map.label, n.map.range)
for (n, m, l) in [(5, 7, 9), (4, 3, 8), (1, 2, 3)]:
    A = np.random.rand(n, m, l); B = np.zeros((n, m, l)); y = np.ones((n, m))
    s(A=A, B=B, y=y, N=n, M=m, L=l)
    ref = np.zeros((n, m, l)); ref[:, 1:, :] = A[:, 1:, :] + A[:, :-1, :] * 0.5
    print(n, m, l, "B", np.allclose(B, ref), "y", np.allclose(y, 1 + A.sum(-1)))
