import sys, time
import numpy as np
import dace
from drv import run

N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def k_step2(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[1:N:2]:
        C[i] = A[i] * 3.0

@dace.program
def k_sum(A: dace.float64[N], s: dace.float64[1]):
    for i in dace.map[0:N]:
        s[0] += A[i]

@dace.program
def k_max(A: dace.float64[N], s: dace.float64[1]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            o >> s(1, lambda x, y: max(x, y))[0]
            o = a

@dace.program
def k_2d(A: dace.float64[M, N], C: dace.float64[M, N]):
    for i, j in dace.map[0:M, 0:N]:
        C[i, j] = A[i, j] + A[i, 0]

@dace.program
def k_2d_T(A: dace.float64[N, M], C: dace.float64[M, N]):
    for i, j in dace.map[0:M, 0:N]:
        C[i, j] = A[j, i] * 2.0

@dace.program
def k_masked_write(A: dace.float64[N], C: dace.float64[N]):
    C[A > 0.5] = 7.0

@dace.program
def k_where(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        if A[i] > 0.5:
            C[i] = A[i] * 2.0
        else:
            C[i] = -A[i]

@dace.program
def k_gather(A: dace.float64[N], idx: dace.int64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[idx[i]]

@dace.program
def k_scatter(A: dace.float64[N], idx: dace.int64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[idx[i]] = A[i]

n, m = 21, 11
rng = np.random.default_rng(0)
Af = rng.random(n)
perm = rng.permutation(n).astype(np.int64)
cases = {
 'k_step2': (k_step2, dict(A=Af, C=np.zeros(n)), dict(N=n)),
 'k_sum': (k_sum, dict(A=Af, s=np.ones(1)), dict(N=n)),
 'k_max': (k_max, dict(A=Af, s=np.full(1, 0.3)), dict(N=n)),
 'k_2d': (k_2d, dict(A=rng.random((m, n)), C=np.zeros((m, n))), dict(N=n, M=m)),
 'k_2d_T': (k_2d_T, dict(A=rng.random((n, m)), C=np.zeros((m, n))), dict(N=n, M=m)),
 'k_masked_write': (k_masked_write, dict(A=Af, C=np.zeros(n)), dict(N=n)),
 'k_where': (k_where, dict(A=Af, C=np.zeros(n)), dict(N=n)),
 'k_gather': (k_gather, dict(A=Af, idx=perm, C=np.zeros(n)), dict(N=n)),
 'k_scatter': (k_scatter, dict(A=Af, idx=perm, C=np.zeros(n)), dict(N=n)),
}
rem = sys.argv[1]
widths = tuple(int(x) for x in sys.argv[2].split(','))
for name in (sys.argv[3:] or cases):
    prog, arrs, params = cases[name]
    print('=====', name, rem, widths)
    t = time.time()
    try:
        run(prog, f'{name}_{rem}_{len(widths)}', arrs, params, remainder=rem, widths=widths, exact=False)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:400])
    print('time', round(time.time() - t, 1))
