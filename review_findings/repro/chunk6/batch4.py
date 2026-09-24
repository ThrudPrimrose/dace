import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def k_ndivm(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + N / M

@dace.program
def k_ndiv4(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * (N / 4)

@dace.program
def k_int_scalar_float(A: dace.float64[N], B: dace.int64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + B[0] * 0.5

@dace.program
def k_int_sym_float(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + (N + 1) * 0.5

@dace.program
def k_sqrt_int(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + np.sqrt(N + 1)

n = 21
Af = np.linspace(0, 1, n)
cases = {
 'k_ndivm': (k_ndivm, dict(A=Af, C=np.zeros(n)), dict(N=n, M=4)),
 'k_ndiv4': (k_ndiv4, dict(A=Af, C=np.zeros(n)), dict(N=n)),
 'k_int_scalar_float': (k_int_scalar_float, dict(A=Af, B=np.arange(n, dtype=np.int64) + 3, C=np.zeros(n)), dict(N=n)),
 'k_int_sym_float': (k_int_sym_float, dict(A=Af, C=np.zeros(n)), dict(N=n)),
 'k_sqrt_int': (k_sqrt_int, dict(A=Af, C=np.zeros(n)), dict(N=n)),
}
rem = sys.argv[1]
for name in (sys.argv[2:] or cases):
    prog, arrs, params = cases[name]
    print('=====', name, rem)
    try:
        run(prog, f'{name}_{rem}', arrs, params, remainder=rem, exact=False)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:400])
