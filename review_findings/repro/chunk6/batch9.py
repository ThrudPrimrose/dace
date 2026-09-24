import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')

@dace.program
def k_i_half(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + i * 0.5

@dace.program
def k_i_div2(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * (i / 2)

@dace.program
def k_i_over_n(C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = i / N

@dace.program
def k_i_float(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + dace.float64(i) * 0.5

n = 21
Af = np.linspace(0, 1, n)
cases = {
 'k_i_half': (k_i_half, dict(A=Af, C=np.zeros(n))),
 'k_i_div2': (k_i_div2, dict(A=Af, C=np.zeros(n))),
 'k_i_over_n': (k_i_over_n, dict(C=np.zeros(n))),
 'k_i_float': (k_i_float, dict(A=Af, C=np.zeros(n))),
}
rem = sys.argv[1]
for name in (sys.argv[2:] or cases):
    prog, arrs = cases[name]
    print('=====', name, rem)
    try:
        run(prog, f'{name}_{rem}', arrs, dict(N=n), remainder=rem, exact=False, lift_copy=False)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:500])
