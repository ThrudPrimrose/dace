import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def postloop(A: dace.float64[10, 10], B: dace.float64[10], C: dace.int64[1]):
    for i in range(N):
        for j in range(N):
            A[i, j] = A[i, j] + 1.0
        B[i] = B[i] + 2.0
    C[0] = i

@dace.program
def negstride(A: dace.float64[20]):
    for i in range(N, M, -2):
        A[i] = A[i] + 1.0

which = sys.argv[1]
if which == 'postloop':
    sdfg = postloop.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True, perfect_loop_nesting=False)
    print('free symbols after canonicalize:', sorted(sdfg.free_symbols))
    A = np.zeros((10, 10)); B = np.zeros(10); C = np.zeros(1, dtype=np.int64)
    sdfg.name = 'postloop_nopln'
    sdfg(A=A, B=B, C=C, N=5)
    print('C =', C[0], '(uncanonicalized dace: 5)')
else:
    sdfg = negstride.to_sdfg(simplify=True)
    canonicalize(sdfg, validate=True)
    A = np.zeros(20)
    sdfg(A=A, N=5, M=5)
    print('N=5,M=5 written:', A.nonzero()[0].tolist(), 'numpy: []')
