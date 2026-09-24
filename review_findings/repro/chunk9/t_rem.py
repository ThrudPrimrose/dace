import sys; sys.path.insert(0, '.')
from drv import *
N = dace.symbol('N')
@dace.program
def k_axpy(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] * 2.0 + b[i]
@dace.program
def k_dot(a: dace.float64[N], b: dace.float64[N], s: dace.float64[1]):
    acc = 0.0
    for i in dace.map[0:N]:
        acc += a[i] * b[i]
    s[0] = acc
n = 29
rng = np.random.default_rng(0)
w, rem = sys.argv[1], sys.argv[2]
extra = dict(scalar_remainder_emit='tile_k1') if rem == 'tile_k1' else {}
if rem == 'tile_k1': rem = 'scalar_postamble'
if w == 'axpy': run(k_axpy, dict(a=rng.random(n), b=rng.random(n), N=n), remainder=rem, name=f'axpy_{rem}', **extra)
if w == 'dot': run(k_dot, dict(a=rng.random(n), b=rng.random(n), s=np.zeros(1), N=n), remainder=rem, name=f'dot_{rem}_{len(extra)}', **extra)
