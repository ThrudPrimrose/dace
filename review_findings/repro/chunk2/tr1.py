import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize.loop_to_transpose import LoopToTranspose
from harness import run_pass_checked, canon_prefix
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def wcr_t(A: dace.float64[M, N], B: dace.float64[N, M]):
    for i in range(N):
        for j in range(M):
            with dace.tasklet:
                a << A[j, i]
                b >> B(1, lambda x, y: x + y)[i, j]
                b = a


mode = sys.argv[1]
sdfg = wcr_t.to_sdfg(simplify=True)
sdfg.name = f'tr1_{mode}'
if mode == 'direct':
    run_pass_checked(LoopToTranspose(), sdfg)
elif mode == 'prefix':
    canon_prefix(sdfg, 'loop_to_x')
    run_pass_checked(LoopToTranspose(), sdfg)
else:
    canonicalize(sdfg)
sdfg.validate()
print('libnodes:', [type(n).__name__ for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.LibraryNode)])
n, m = 4, 3
A = np.arange(m * n, dtype=np.float64).reshape(m, n).copy()
B = np.ones((n, m))
sdfg(A=A, B=B, N=n, M=m)
print('B ok:', np.allclose(B, 1 + A.T))
print(B)
