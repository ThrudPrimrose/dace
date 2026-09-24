import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
N = dace.symbol('N')

@dace.program
def k_i_half(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + i * 0.5

s = k_i_half.to_sdfg(simplify=True)
canonicalize(s, validate=True)
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet): print('BEFORE', repr(n.code.as_string))
VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR, remainder_strategy='scalar_postamble', expand_tile_nodes=False)).apply_pass(s, {})
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.LibraryNode):
        outs = [e.data.data for e in g.out_edges(n)]
        ins = [(e.dst_conn, e.data.data, str(g.sdfg.arrays[e.data.data].dtype), g.sdfg.arrays[e.data.data].shape) for e in g.in_edges(n) if e.data.data]
        print('NODE', type(n).__name__, n.label, getattr(n, 'op', None), getattr(n,'kind_a',None), getattr(n,'expr_a',None), getattr(n,'kind_b',None), getattr(n,'expr_b',None), 'in', ins, 'out', outs, [str(g.sdfg.arrays[o].dtype) for o in outs])
    if isinstance(n, dace.nodes.Tasklet): print('T', n.label, n.code.as_string[:300])
s.expand_library_nodes()
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet) and 'binop' in n.label: print('EXPANDED', n.label, n.code.as_string.splitlines()[-1])
