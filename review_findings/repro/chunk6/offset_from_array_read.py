"""``C[i] = A[off[0] + i]``: the index symbol hides the iter-var behind a data-reading interstate
assignment. The classifier says LINEAR, the source memlet stays one element wide, and
InsertTileLoadStore stages a stride-1 Tile TileLoad over a by-value source -> lane 0 is splatted."""
import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')


@dace.program
def shifted_copy(A: dace.float64[2 * N], off: dace.int64[1], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[off[0] + i]


n = 21
rem = sys.argv[1] if len(sys.argv) > 1 else 'scalar_postamble'
A = np.arange(2 * n, dtype=np.float64)
print('numpy expects', A[3:3 + n][:10])
run(shifted_copy, f'shifted_copy_{rem}', dict(A=A, off=np.array([3], np.int64), C=np.zeros(n)), dict(N=n),
    remainder=rem, exact=False)
