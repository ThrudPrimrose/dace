import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize.materialize_loop_exit_symbols import MaterializeLoopExitSymbols
from harness import run_pass_checked, canon_prefix, run_until
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def post_update(a: dace.int64[N], b: dace.int64[2]):
    for i in range(N):
        a[i] = i
    i = i + 5
    b[0] = i
    b[1] = i * 3


mode = sys.argv[1]
sdfg = post_update.to_sdfg(simplify=True)
sdfg.name = f'mx4_{mode}'
if mode == 'direct':
    run_pass_checked(MaterializeLoopExitSymbols(), sdfg)
elif mode == 'prefix':
    run_until(sdfg, MaterializeLoopExitSymbols)
    sdfg.save('mx4_prefix.sdfg')
    run_pass_checked(MaterializeLoopExitSymbols(), sdfg)
    sdfg.save('mx4_after.sdfg')
else:
    canonicalize(sdfg)
sdfg.validate()
n = 6
a = np.zeros(n, dtype=np.int64)
b = np.zeros(2, dtype=np.int64)
sdfg(a=a, b=b, N=n)
print('a', a, 'b', b, 'python-expected b', [n - 1 + 5, 3 * (n - 1 + 5)])
