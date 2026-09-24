import sys; sys.path.insert(0, '.')
from drv import *
N = dace.symbol('N')
@dace.program
def k_g(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[idx[i]]
@dace.program
def k_g2(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N]):
    for i in dace.map[0:N-1]:
        b[i] = a[idx[i + 1]] + a[idx[i]]
@dace.program
def k_s(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[idx[i]] = a[i] * 2.0
@dace.program
def k_gr(a: dace.float64[N], idx: dace.int32[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[idx[N - 1 - i]]
n = 29
rng = np.random.default_rng(0)
perm = rng.permutation(n).astype(np.int32)
w = sys.argv[1]
p = dict(g=k_g, g2=k_g2, s=k_s, gr=k_gr)[w]
run(p, dict(a=rng.random(n), idx=perm, b=np.zeros(n), N=n), show=True, loop_to_map_permissive=(w=='s'))
