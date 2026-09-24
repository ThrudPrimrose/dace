import sys; sys.path.insert(0, '.')
from drv import *
N = dace.symbol('N'); M = dace.symbol('M')
@dace.program
def k_g2d(a: dace.float64[N * M], idx: dace.int32[N, M], b: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[idx[i, j]]
@dace.program
def k_grow(a: dace.float64[N, M], idx: dace.int32[N], b: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[idx[i], j]
n, m = 11, 13
rng = np.random.default_rng(0)
w = sys.argv[1]
if w == 'g2d': run(k_g2d, dict(a=rng.random(n*m), idx=rng.permutation(n*m).reshape(n, m).astype(np.int32), b=np.zeros((n, m)), N=n, M=m), widths=(4, 4), show=True)
if w == 'grow': run(k_grow, dict(a=rng.random((n, m)), idx=rng.permutation(n).astype(np.int32), b=np.zeros((n, m)), N=n, M=m), widths=(4, 4), show=True)
