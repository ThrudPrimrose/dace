import numpy as np, dace, copy
from drv import *
N = 16
@dace.program
def f(A: dace.int32[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = A[i] ** 2.0
s = f.to_sdfg(simplify=True)
ref_s = copy.deepcopy(s)
vec(s)
print("tile ops", ntile(s))
for n,_ in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet): print(repr(n.code.as_string))
A = np.full(N, 100000, np.int32)
B = np.zeros(N); B0 = np.zeros(N)
ref_s.name = 'ref_powint'
ref_s(A=A, B=B0)
s(A=A, B=B)
print("numpy", (A**2.0)[:3], "unvectorized dace", B0[:3], "vectorized", B[:3])
