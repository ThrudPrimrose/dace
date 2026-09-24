import sys
import numpy as np
import dace
from drv import run
N = dace.symbol('N')

@dace.program
def k_ramp_only(C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = i * 0.5

@dace.program
def k_int_scalar_arg(A: dace.float64[N], k: dace.int64, C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + k * 0.5

n = 21
for prog, arrs, params in ((k_ramp_only, dict(C=np.zeros(n)), dict(N=n)),
                           (k_int_scalar_arg, dict(A=np.linspace(0, 1, n), C=np.zeros(n)), dict(N=n, k=3))):
    print('==', prog.name)
    try:
        run(prog, f'{prog.name}_b10', arrs, params, remainder='scalar_postamble', exact=False, lift_copy=False)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:300])
