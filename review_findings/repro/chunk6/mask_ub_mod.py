"""TileMaskGen's bound is ``str(ub + 1)`` of a sympy expression, emitted verbatim into C++ (Mod)."""
import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def mod_bound(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N - (M % 4)]:
        C[i] = A[i] * 2.0


n = 21
rem = sys.argv[1] if len(sys.argv) > 1 else 'full_mask'
A = np.linspace(0.0, 1.0, n)
run(mod_bound, f'mod_bound_{rem}', dict(A=A, C=np.zeros(n)), dict(N=n, M=7), remainder=rem, exact=False)
