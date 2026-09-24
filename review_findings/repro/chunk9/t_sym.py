import sys; sys.path.insert(0, '.')
from drv import *
N = dace.symbol('N')
@dace.program
def k_sym(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        k = i + 1
        if a[i] > 0.5:
            b[i] = k * 2.0
        else:
            b[i] = k * 0.5
@dace.program
def k_sym2(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        k = i * 3
        b[i] = a[i] + k
n = 29
rng = np.random.default_rng(0)
w = sys.argv[1]
p = dict(sym=k_sym, sym2=k_sym2)[w]
run(p, dict(a=rng.random(n), b=np.zeros(n), N=n), show=True)
