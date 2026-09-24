import copy
import numpy as np
import dace
from dace.transformation.interstate import MapLoopInterchange
N, T = dace.symbol('N'), dace.symbol('T')

@dace.program
def prog(A: dace.float64[T, N], s: dace.float64[1], M: dace.float64[N]):
    for i in dace.map[0:N]:
        for t in range(T):
            s[0] += A[t, i]
            M[i] = max(M[i], A[t, i]) * 0.5 + s[0] * 0.0

@dace.program
def prog2(A: dace.float64[T, N], B: dace.float64[N]):
    for i, j in dace.map[0:N, 0:N]:
        for t in range(T - 1, 0, -2):
            B[i] += A[t, j] * (i + 1)

for p, args in [(prog, lambda: dict(A=np.arange(12.).reshape(3, 4).copy(), s=np.zeros(1), M=np.zeros(4), N=4, T=3)),
                (prog2, lambda: dict(A=np.arange(20.).reshape(5, 4).copy(), B=np.zeros(4), N=4, T=5))]:
    sdfg = p.to_sdfg(simplify=True)
    ref = copy.deepcopy(sdfg)
    print(p.name, 'applied', sdfg.apply_transformations(MapLoopInterchange))
    a, b = args(), args()
    ref(**a); sdfg(**b)
    print({k: (np.allclose(a[k], b[k]), a[k], b[k]) for k in a if isinstance(a[k], np.ndarray)})
