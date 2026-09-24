"""``C[i] = A[idx[i] + 1] * 2.0`` under masked_tail: WidenAccesses' per-lane symbol fan-out prints the
index as ``Subscript(idx, ...)`` into an interstate assignment, which does not compile."""
import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')


@dace.program
def gather_plus_one(A: dace.float64[N + 1], idx: dace.int64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[idx[i] + 1] * 2.0


n = 21
rem = sys.argv[1] if len(sys.argv) > 1 else 'masked_tail'
rng = np.random.default_rng(1)
try:
    run(gather_plus_one, f'gather_plus_one_{rem}', dict(A=rng.random(n + 1), idx=rng.permutation(n).astype(np.int64),
                                                        C=np.zeros(n)), dict(N=n), remainder=rem, exact=False)
except Exception as e:
    msg = str(e)
    print('EXCEPTION', type(e).__name__, [l for l in msg.splitlines() if 'error' in l][:2])
