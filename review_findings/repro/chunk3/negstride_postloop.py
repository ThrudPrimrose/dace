import numpy as np
import dace
from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride

N = dace.symbol('N')

@dace.program
def prog(A: dace.float64[20], B: dace.int64[1]):
    for i in range(N, 2, -1):
        A[i] = A[i] + 1.0
    B[0] = i

sdfg = prog.to_sdfg(simplify=True)
A = np.zeros(20); B = np.zeros(1, dtype=np.int64)
sdfg(A=A, B=B, N=10)
print('before pass: B =', B[0])
print('pass returned', NormalizeNegativeStride().apply_pass(sdfg, {}))
A = np.zeros(20); B = np.zeros(1, dtype=np.int64)
sdfg(A=A, B=B, N=10)
print('after pass: B =', B[0], ' python i after loop = 3')
