import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize.loop_to_einsum import LoopToEinsum
from harness import run_pass_checked, canon_prefix
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def part_t(A: dace.float64[M, M], B: dace.float64[M, M]):
    for i in range(N):
        for j in range(N):
            B[i, j] = A[j, i]


mode = sys.argv[1]
sdfg = part_t.to_sdfg(simplify=True)
sdfg.name = f'ein1_{mode}'
if mode == 'direct':
    run_pass_checked(LoopToEinsum(), sdfg)
elif mode == 'prefix':
    canon_prefix(sdfg, 'loop_to_x')
    run_pass_checked(LoopToEinsum(), sdfg)
elif mode == 'full_nosem':
    canonicalize(sdfg, semantic_lifting=False)
else:
    canonicalize(sdfg)
sdfg.validate()
print('libnodes:', [type(n).__name__ for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.LibraryNode)])
n, m = 3, 5
A = np.arange(m * m, dtype=np.float64).reshape(m, m).copy()
B = -np.ones((m, m))
sdfg(A=A, B=B, N=n, M=m)
ref = -np.ones((m, m))
ref[:n, :n] = A[:n, :n].T
print('B ok:', np.allclose(B, ref))
print(B)
