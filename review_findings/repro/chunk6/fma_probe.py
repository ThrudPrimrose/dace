import copy
import numpy as np
import dace
from dace.transformation.passes.vectorization.fuse_multiply_add import FuseMultiplyAdd
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')

@dace.program
def k(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N], D: dace.float64[N]):
    t = dace.define_local_scalar(dace.float64)
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            b << B[i]
            o >> t
            o = a * b
        with dace.tasklet:
            x << t
            a2 << A[i]
            c >> C[i]
            c = x + a2
    for j in range(1, N):
        with dace.tasklet:
            d << D[j - 1]
            o2 >> t
            o2 = d * 2.0
        with dace.tasklet:
            y << t
            e >> D[j]
            e = y + 1.0

sdfg = k.to_sdfg(simplify=True)
sdfg.save('fma_in.sdfg')
from dace.sdfg import nodes
for n, st in sdfg.all_nodes_recursive():
    if isinstance(n, nodes.AccessNode): print(st.parent_graph.__class__.__name__, st.label, n.data, st.in_degree(n), st.out_degree(n))
print(FuseMultiplyAdd().apply_pass(sdfg, {}))
sdfg.validate()
