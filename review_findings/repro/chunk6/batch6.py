import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')

@dace.program
def k_off_write(A: dace.float64[N], off: dace.int64[1], C: dace.float64[2 * N]):
    for i in dace.map[0:N]:
        C[off[0] + i] = A[i]

@dace.program
def k_off_read(A: dace.float64[2 * N], off: dace.int64[1], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[off[0] + i]

@dace.program
def k_scatter_arith(A: dace.float64[N], idx: dace.int64[N], C: dace.float64[N + 1]):
    for i in dace.map[0:N]:
        C[idx[i] + 1] = A[i]

@dace.program
def k_gather_arith(A: dace.float64[N + 1], idx: dace.int64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[idx[i] + 1] * 2.0

@dace.program
def k_gather2(A: dace.float64[N], idx: dace.int64[N], jdx: dace.int64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[idx[i]] + A[jdx[i]]

n = 21
rng = np.random.default_rng(1)
Af = rng.random(n)
perm = rng.permutation(n).astype(np.int64)
perm2 = rng.permutation(n).astype(np.int64)
cases = {
 'k_off_write': (k_off_write, dict(A=Af, off=np.array([3], np.int64), C=np.zeros(2 * n))),
 'k_off_read': (k_off_read, dict(A=rng.random(2 * n), off=np.array([3], np.int64), C=np.zeros(n))),
 'k_scatter_arith': (k_scatter_arith, dict(A=Af, idx=perm, C=np.zeros(n + 1))),
 'k_gather_arith': (k_gather_arith, dict(A=rng.random(n + 1), idx=perm, C=np.zeros(n))),
 'k_gather2': (k_gather2, dict(A=Af, idx=perm, jdx=perm2, C=np.zeros(n))),
}
rem = sys.argv[1]
for name in (sys.argv[2:] or cases):
    prog, arrs = cases[name]
    print('=====', name, rem)
    try:
        run(prog, f'{name}_{rem}', arrs, dict(N=n), remainder=rem, exact=False)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:500])
