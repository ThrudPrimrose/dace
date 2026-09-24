"""FuseMultiplyAdd fuses an INTEGER a*b + c into std::fma, which computes in double."""
import numpy as np
import dace
from drv import run

N = dace.symbol('N')


@dace.program
def axpy_i64(A: dace.int64[N], B: dace.int64[N], C: dace.int64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * B[i] + A[i]


n = 16
A = np.full(n, 300_000_001, dtype=np.int64)
B = np.full(n, 300_000_007, dtype=np.int64)
run(axpy_i64, 'fma_int64', dict(A=A, B=B, C=np.zeros(n, np.int64)), dict(N=n), fma=True)
