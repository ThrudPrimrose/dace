import dace, numpy as np
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
N = dace.symbol('N')

@dace.program
def k_ndiv4(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] * (N / 4)

s = k_ndiv4.to_sdfg(simplify=True)
canonicalize(s, validate=True)
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet): print('BEFORE', repr(n.code.as_string))
VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR, remainder_strategy='scalar_postamble', expand_tile_nodes=False)).apply_pass(s, {})
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.LibraryNode) and 'Binop' in type(n).__name__:
        outs = [e.data.data for e in g.out_edges(n)]
        print('NODE', n.label, n.op, n.kind_a, n.expr_a, n.kind_b, n.expr_b, 'out', outs, g.sdfg.arrays[outs[0]].dtype)
s.expand_library_nodes()
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet) and 'binop' in n.label: print('EXPANDED', n.label, n.code.as_string.splitlines()[-1])
