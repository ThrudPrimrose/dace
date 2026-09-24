import sys; sys.path.insert(0, '.')
from drv import *
N = dace.symbol('N'); M = dace.symbol('M'); S = dace.symbol('S')
@dace.program
def k_symstride(a: dace.float64[N * M], b: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[j * N + i]
@dace.program
def k_symstride_w(a: dace.float64[N, M], b: dace.float64[N * M]):
    for i, j in dace.map[0:N, 0:M]:
        b[j * N + i] = a[i, j] + 1.0
@dace.program
def k_inc(a: dace.float64[N * S], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i * S + 1]
n, m = 13, 19
rng = np.random.default_rng(0)
w = sys.argv[1]
if w == 'ss': run(k_symstride, dict(a=rng.random(n*m), b=np.zeros((n, m)), N=n, M=m), widths=(8,), show=True)
if w == 'ssw': run(k_symstride_w, dict(a=rng.random((n, m)), b=np.zeros(n*m), N=n, M=m), widths=(8,), show=True)
if w == 'inc': run(k_inc, dict(a=rng.random(n*3), b=np.zeros(n), N=n, S=3), widths=(8,), show=True)
