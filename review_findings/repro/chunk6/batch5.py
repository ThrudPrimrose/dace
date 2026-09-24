import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')

@dace.program
def k_intdiv_lit(B: dace.int64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = B[i] / 2

@dace.program
def k_intdiv_sym(B: dace.int64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = B[i] / N

@dace.program
def k_int_plus_half(B: dace.int32[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = B[i] + 0.5

n = 21
B = np.arange(n, dtype=np.int64) + 1
cases = {
 'k_intdiv_lit': (k_intdiv_lit, dict(B=B, C=np.zeros(n)), dict(N=n)),
 'k_intdiv_sym': (k_intdiv_sym, dict(B=B, C=np.zeros(n)), dict(N=n)),
 'k_int_plus_half': (k_int_plus_half, dict(B=B.astype(np.int32), C=np.zeros(n)), dict(N=n)),
}
rem = sys.argv[1]
for name in (sys.argv[2:] or cases):
    prog, arrs, params = cases[name]
    print('=====', name, rem)
    try:
        run(prog, f'{name}_{rem}', arrs, params, remainder=rem, exact=False)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:400])
