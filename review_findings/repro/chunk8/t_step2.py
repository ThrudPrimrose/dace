import numpy as np, dace, copy, sys
from drv import *
N = 32
@dace.program
def f(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N:2]:
        B[i] = A[i] + 1.0
s = f.to_sdfg(simplify=True)
vec(s, remainder_strategy=sys.argv[1] if len(sys.argv) > 1 else 'masked_tail')
print("tile ops", ntile(s))
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.MapEntry): print(n.map.label, n.map.range)
A = np.arange(N, dtype=np.float64); B = np.zeros(N)
s(A=A, B=B)
ref = np.zeros(N); ref[::2] = A[::2] + 1
print("match", np.array_equal(B, ref), B[:6])
