import numpy as np
import dace
from dace.transformation.passes.canonicalize.normalize_loops_and_maps import NormalizeStridedMaps

N = dace.symbol('N')

@dace.program
def prog(A: dace.float64[20]):
    for i in dace.map[3:N:2]:
        A[i] = A[i] + 1.0

sdfg = prog.to_sdfg(simplify=True)
me = next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.MapEntry))
print('before:', me.map.range)
print('pass returned', NormalizeStridedMaps().apply_pass(sdfg, {}))
print('after :', me.map.range)
A = np.zeros(20); ref = np.zeros(20)
for i in range(3, 3, 2):
    ref[i] += 1
sdfg(A=A, N=3)
print('N=3 dace written:', A.nonzero()[0].tolist(), ' numpy written:', ref.nonzero()[0].tolist())
