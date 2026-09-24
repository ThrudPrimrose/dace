import numpy as np
import dace
from dace.transformation.passes.canonicalize.normalize_loop_and_map_origin import NormalizeLoopAndMapOrigin

N = dace.symbol('N')

@dace.program
def prog(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[1:N]:
        with dace.tasklet:
            a << A[i]
            b >> B[i]
            t = a * 2.0
            b = t + 1.0

sdfg = prog.to_sdfg(simplify=True)
tasklets = lambda: [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.Tasklet)]
print('before:', [t.code.as_string for t in tasklets()])
print('pass returned', NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}))
print('after:', [t.code.as_string for t in tasklets()])
A = np.arange(8, dtype=np.float64); B = np.zeros(8)
sdfg(A=A, B=B, N=8)
ref = np.zeros(8); ref[1:] = A[1:] * 2 + 1
print('dace :', B.tolist()); print('numpy:', ref.tolist())
