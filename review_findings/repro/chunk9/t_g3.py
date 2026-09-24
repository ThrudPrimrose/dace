import sys; sys.path.insert(0, '.')
from drv import *
N = dace.symbol('N')
@dace.program
def k_ga(a: dace.float64[N + 1], idx: dace.int32[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[idx[i] + 1]
@dace.program
def k_gb(a: dace.float64[N + 1], idx: dace.int32[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        k = idx[i] + 1
        b[i] = a[k]
@dace.program
def k_gc(a: dace.float64[2 * N], idx: dace.int32[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        k = idx[i] * 2
        b[i] = a[k] + a[k + 1]
@dace.program
def k_gd(a: dace.float64[N + 1], idx: dace.int32[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        k = idx[i]
        b[k] = a[i] * 3.0
n = 29
rng = np.random.default_rng(0)
perm = rng.permutation(n).astype(np.int32)
w = sys.argv[1]
p = dict(ga=k_ga, gb=k_gb, gc=k_gc, gd=k_gd)[w]
run(p, dict(a=rng.random(2*n), idx=perm, b=np.zeros(n), N=n) if w=='gc' else dict(a=rng.random(n+1), idx=perm, b=np.zeros(n), N=n), show=True, loop_to_map_permissive=(w=='gd'))
