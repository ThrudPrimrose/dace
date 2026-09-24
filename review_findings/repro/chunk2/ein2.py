import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize.loop_to_einsum import LoopToEinsum
from harness import run_pass_checked, canon_prefix
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def part_mul1(A: dace.float64[M, M], B: dace.float64[M, M]):
    for i in range(N):
        for j in range(N):
            B[i, j] = 1.0 * A[j, i]


@dace.program
def part_tasklet(A: dace.float64[M, M], B: dace.float64[M, M]):
    for i in range(N):
        for j in range(N):
            with dace.tasklet:
                a << A[j, i]
                b >> B[i, j]
                b = a


mode = sys.argv[1]
variant = sys.argv[2]
sdfg = {'mul1': part_mul1, 'tasklet': part_tasklet}[variant].to_sdfg(simplify=True)
sdfg.name = f'ein2_{mode}_{variant}'
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
