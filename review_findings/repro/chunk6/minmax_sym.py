"""0-in-connector symbol body ``min(N, M) + max(N, M)``: the textual two-symbol splitter can bind
``min(`` to the whole RHS; which op wins depends on set iteration order (PYTHONHASHSEED)."""
import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def minmax(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + (min(N, M) + max(N, M))


n = 21
A = np.linspace(0.0, 1.0, n)
print('numpy', (A + (min(n, 5) + max(n, 5)))[:3])
run(minmax, f'minmax_{sys.argv[1]}', dict(A=A, C=np.zeros(n)), dict(N=n, M=5), remainder='scalar_postamble', exact=False)
