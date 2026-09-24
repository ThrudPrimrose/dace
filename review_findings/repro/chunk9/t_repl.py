import sys; sys.path.insert(0, '.')
from drv import *
N = dace.symbol('N')
@dace.program
def repl_off(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N-1]:
        b[i] = a[(i + 1) // 2]
@dace.program
def repl_coef(a: dace.float64[2*N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[(3 * i) // 2]
n = 32
which = sys.argv[1]
if which == 'off':
    run(repl_off, dict(a=np.arange(n, dtype=np.float64), b=np.zeros(n), N=n), show=True)
else:
    run(repl_coef, dict(a=np.arange(2*n, dtype=np.float64), b=np.zeros(n), N=n), show=True)
