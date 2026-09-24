import sys; sys.path.insert(0, '.')
from drv import *
N = dace.symbol('N')
@dace.program
def k_rev(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[N - 1 - i]
@dace.program
def k_revw(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[N - 1 - i] = a[i] + 1.0
@dace.program
def k_intdiv(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[int(i / 2)]
@dace.program
def k_sq(a: dace.float64[N*N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i * i]
@dace.program
def k_mod(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[(i + 3) % 5]
@dace.program
def k_half(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i // 2]
n = 37
progs = dict(rev=(k_rev, n), revw=(k_revw, n), intdiv=(k_intdiv, n), sq=(k_sq, n*n), mod=(k_mod, n), half=(k_half, n))
p, na = progs[sys.argv[1]]
run(p, dict(a=np.arange(na, dtype=np.float64), b=np.zeros(n), N=n), show=True)
