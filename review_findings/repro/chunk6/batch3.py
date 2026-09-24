import sys, time, math
import numpy as np
import dace
from drv import run

N = dace.symbol('N')

@dace.program
def k_symprod(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * (N - 1)

@dace.program
def k_sqrtN(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = math.sqrt(N) + A[i]

@dace.program
def k_isq(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + i * i

@dace.program
def k_modgather(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[(i * 3) % N]

@dace.program
def k_powN(A: dace.int64[N], C: dace.int64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] ** 3

@dace.program
def k_shift(A: dace.int64[N], C: dace.int64[N]):
    for i in dace.map[0:N]:
        C[i] = (A[i] >> 1) + (A[i] & 7)

@dace.program
def k_ite_symarm(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        if A[i] > 0.5:
            C[i] = A[i]
        else:
            C[i] = N

@dace.program
def k_ite_lanearm(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        if A[i] > 0.5:
            C[i] = A[i]
        else:
            C[i] = i

@dace.program
def k_ite_iv_cond(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        if i % 2 == 0:
            C[i] = A[i]
        else:
            C[i] = -A[i]

@dace.program
def k_neg_sym(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = -N + A[i]

@dace.program
def k_nsq(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] - (N * N) / 7

n = 21
rng = np.random.default_rng(0)
Af = rng.random(n)
Ai = rng.integers(-50, 50, n).astype(np.int64)
cases = {
 'k_symprod': (k_symprod, dict(A=Af, C=np.zeros(n))),
 'k_sqrtN': (k_sqrtN, dict(A=Af, C=np.zeros(n))),
 'k_isq': (k_isq, dict(A=Af, C=np.zeros(n))),
 'k_modgather': (k_modgather, dict(A=Af, C=np.zeros(n))),
 'k_powN': (k_powN, dict(A=Ai, C=np.zeros(n, np.int64))),
 'k_shift': (k_shift, dict(A=Ai, C=np.zeros(n, np.int64))),
 'k_ite_symarm': (k_ite_symarm, dict(A=Af, C=np.zeros(n))),
 'k_ite_lanearm': (k_ite_lanearm, dict(A=Af, C=np.zeros(n))),
 'k_ite_iv_cond': (k_ite_iv_cond, dict(A=Af, C=np.zeros(n))),
 'k_neg_sym': (k_neg_sym, dict(A=Af, C=np.zeros(n))),
 'k_nsq': (k_nsq, dict(A=Af, C=np.zeros(n))),
}
rem = sys.argv[1]
for name in (sys.argv[2:] or cases):
    prog, arrs = cases[name]
    print('=====', name, rem)
    try:
        run(prog, f'{name}_{rem}', arrs, dict(N=n), remainder=rem, exact=False)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:400])
