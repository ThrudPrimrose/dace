"""TileMaskGen's bound is ``str(ub + 1)`` of a sympy expression, emitted verbatim into C++."""
import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')


@dace.program
def flat_square(A: dace.float64[N * N], C: dace.float64[N * N]):
    for i in dace.map[0:N * N]:
        C[i] = A[i] * 2.0


n = 5
rem = sys.argv[1] if len(sys.argv) > 1 else 'masked_tail'
A = np.linspace(0.0, 1.0, n * n)
run(flat_square, f'flat_square_{rem}', dict(A=A, C=np.zeros(n * n)), dict(N=n), remainder=rem, exact=False)
