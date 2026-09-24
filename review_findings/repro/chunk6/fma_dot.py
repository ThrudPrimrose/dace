"""FuseMultiplyAdd on reductions: s += A[i] * B[i] with fuse_multiply_add=True."""
import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')


@dace.program
def dot(A: dace.float64[N], B: dace.float64[N], s: dace.float64[1]):
    for i in dace.map[0:N]:
        s[0] += A[i] * B[i]


@dace.program
def rowdot(A: dace.float64[N, N], B: dace.float64[N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        acc = 0.0
        for j in range(N):
            acc = acc + A[i, j] * B[j]
        out[i] = acc


@dace.program
def axpy_acc(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = C[i] + A[i] * B[i]


n = 21
rem = sys.argv[1] if len(sys.argv) > 1 else 'masked_tail'
rng = np.random.default_rng(0)
A = rng.random(n); B = rng.random(n)
for prog, arrs in ((dot, dict(A=A, B=B, s=np.ones(1))), (rowdot, dict(A=rng.random((n, n)), B=B, out=np.zeros(n))),
                   (axpy_acc, dict(A=A, B=B, C=rng.random(n)))):
    print('==', prog.name)
    try:
        run(prog, f'fma_{prog.name}_{rem}', arrs, dict(N=n), remainder=rem, exact=False, fma=True)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:300])
