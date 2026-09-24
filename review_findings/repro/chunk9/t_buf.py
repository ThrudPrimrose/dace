import sys; sys.path.insert(0, '.')
from drv import *
N = dace.symbol('N')
@dace.program
def k_buf(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        tmp = np.empty(2, dtype=np.float64)
        tmp[0] = a[i]
        tmp[1] = a[i] * 2.0
        b[i] = tmp[0] + tmp[1]
@dace.program
def k_win(a: dace.float64[N + 1], b: dace.float64[N]):
    for i in dace.map[0:N]:
        tmp = np.empty(2, dtype=np.float64)
        tmp[:] = a[i:i + 2]
        b[i] = tmp[0] * tmp[1]
n = 29
rng = np.random.default_rng(0)
w = sys.argv[1]
if w == 'buf': run(k_buf, dict(a=rng.random(n), b=np.zeros(n), N=n), show=True)
if w == 'win': run(k_win, dict(a=rng.random(n+1), b=np.zeros(n), N=n), show=True)
