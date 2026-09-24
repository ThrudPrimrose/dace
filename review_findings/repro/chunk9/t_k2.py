import sys; sys.path.insert(0, '.')
from drv import *
N = dace.symbol('N'); M = dace.symbol('M')
@dace.program
def k_tr(a: dace.float64[M, N], b: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[j, i]
@dace.program
def k_rep(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[i // 2, j]
@dace.program
def k_sum(a: dace.float64[N + M], b: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[i + j]
@dace.program
def k_bc(a: dace.float64[N], c: dace.float64[M], b: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[i] * c[j]
@dace.program
def k_rev2(a: dace.float64[N, M], b: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[N - 1 - i, M - 1 - j]
n, m = 19, 21
w = sys.argv[1]
if w == 'tr': run(k_tr, dict(a=np.random.rand(m, n), b=np.zeros((n, m)), N=n, M=m), widths=(4, 4), show=True)
if w == 'rep': run(k_rep, dict(a=np.random.rand(n, m), b=np.zeros((n, m)), N=n, M=m), widths=(4, 4), show=True)
if w == 'sum': run(k_sum, dict(a=np.random.rand(n + m), b=np.zeros((n, m)), N=n, M=m), widths=(4, 4), show=True)
if w == 'bc': run(k_bc, dict(a=np.random.rand(n), c=np.random.rand(m), b=np.zeros((n, m)), N=n, M=m), widths=(4, 4), show=True)
if w == 'rev2': run(k_rev2, dict(a=np.random.rand(n, m), b=np.zeros((n, m)), N=n, M=m), widths=(4, 4), show=True)
