import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')

@dace.program
def k_copy_shift(A: dace.float64[N], C: dace.float64[N + 3]):
    for i in dace.map[0:N]:
        C[i + 3] = A[i]

@dace.program
def k_copy_col(A: dace.float64[N], C: dace.float64[N, 4]):
    for i in dace.map[0:N]:
        C[i, 1] = A[i]

@dace.program
def k_copy_stride(A: dace.float64[N], C: dace.float64[2 * N]):
    for i in dace.map[0:N]:
        C[2 * i] = A[i]

@dace.program
def k_copy_rev(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[N - 1 - i] = A[i]

@dace.program
def k_copy_read_shift(A: dace.float64[N + 3], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i + 3]

n = 21
Af = np.arange(n, dtype=np.float64) + 1
cases = {
 'k_copy_shift': (k_copy_shift, dict(A=Af, C=np.zeros(n + 3))),
 'k_copy_col': (k_copy_col, dict(A=Af, C=np.zeros((n, 4)))),
 'k_copy_stride': (k_copy_stride, dict(A=Af, C=np.zeros(2 * n))),
 'k_copy_rev': (k_copy_rev, dict(A=Af, C=np.zeros(n))),
 'k_copy_read_shift': (k_copy_read_shift, dict(A=np.arange(n + 3, dtype=np.float64), C=np.zeros(n))),
}
rem = sys.argv[1]
for name in (sys.argv[2:] or cases):
    prog, arrs = cases[name]
    print('=====', name, rem)
    try:
        run(prog, f'{name}_{rem}', arrs, dict(N=n), remainder=rem, exact=False, lift_copy=False)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:500])
