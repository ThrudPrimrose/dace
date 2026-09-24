import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')

@dace.program
def k_bcast_shift(B: dace.float64[2], C: dace.float64[N + 3]):
    for i in dace.map[0:N]:
        C[i + 3] = B[1]

@dace.program
def k_bcast_col(B: dace.float64[2], C: dace.float64[N, 4]):
    for i in dace.map[0:N]:
        C[i, 1] = B[1]

@dace.program
def k_bcast_stride(B: dace.float64[2], C: dace.float64[2 * N]):
    for i in dace.map[0:N]:
        C[2 * i] = B[1]

n = 21
B = np.array([5.0, 7.0])
cases = {
 'k_bcast_shift': (k_bcast_shift, dict(B=B, C=np.zeros(n + 3))),
 'k_bcast_col': (k_bcast_col, dict(B=B, C=np.zeros((n, 4)))),
 'k_bcast_stride': (k_bcast_stride, dict(B=B, C=np.zeros(2 * n))),
}
rem = sys.argv[1]
for name in (sys.argv[2:] or cases):
    prog, arrs = cases[name]
    print('=====', name, rem)
    try:
        run(prog, f'{name}_{rem}', arrs, dict(N=n), remainder=rem, exact=False, lift_copy=False)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:500])
