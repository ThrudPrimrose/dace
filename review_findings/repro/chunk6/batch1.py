import sys, time, traceback
import numpy as np
import dace
from drv import run

N = dace.symbol('N')

@dace.program
def k_floordiv_const(C: dace.int64[N]):
    for i in dace.map[0:N]:
        C[i] = N // 2

@dace.program
def k_negmod(A: dace.int64[N], C: dace.int64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] % 3

@dace.program
def k_negfloordiv(A: dace.int64[N], C: dace.int64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] // 3

@dace.program
def k_itervar_ite(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] if i < 5 else 0.0

@dace.program
def k_iter_mod(C: dace.int64[N]):
    for i in dace.map[0:N]:
        C[i] = i % 3

@dace.program
def k_rev(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[N - 1 - i]

@dace.program
def k_stencil(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[1:N - 1]:
        C[i] = A[i - 1] + A[i] + A[i + 1]

@dace.program
def k_twowrites(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i]
        C[i] = C[i] * 2.0

n = 21
rng = np.random.default_rng(0)
Ai = rng.integers(-50, 50, n).astype(np.int64)
Af = rng.random(n)
cases = {
 'k_floordiv_const': (k_floordiv_const, dict(C=np.zeros(n, np.int64))),
 'k_negmod': (k_negmod, dict(A=Ai, C=np.zeros(n, np.int64))),
 'k_negfloordiv': (k_negfloordiv, dict(A=Ai, C=np.zeros(n, np.int64))),
 'k_itervar_ite': (k_itervar_ite, dict(A=Af, C=np.zeros(n))),
 'k_iter_mod': (k_iter_mod, dict(C=np.zeros(n, np.int64))),
 'k_rev': (k_rev, dict(A=Af, C=np.zeros(n))),
 'k_stencil': (k_stencil, dict(A=Af, C=np.zeros(n))),
 'k_twowrites': (k_twowrites, dict(A=Af, C=np.zeros(n))),
}
rem = sys.argv[1]
for name in (sys.argv[2:] or cases):
    prog, arrs = cases[name]
    print('=====', name, rem)
    t = time.time()
    try:
        run(prog, f'{name}_{rem}', arrs, dict(N=n), remainder=rem)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:400])
    print('time', round(time.time() - t, 1))
