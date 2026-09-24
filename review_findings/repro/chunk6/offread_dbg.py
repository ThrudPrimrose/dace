import sys, copy
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
N = dace.symbol('N')

@dace.program
def k_off_read(A: dace.float64[2 * N], off: dace.int64[1], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[off[0] + i]

@dace.program
def k_gather_arith(A: dace.float64[N + 1], idx: dace.int64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[idx[i] + 1] * 2.0

prog = {'off': k_off_read, 'ga': k_gather_arith}[sys.argv[1]]
s = prog.to_sdfg(simplify=True)
canonicalize(s, validate=True)
def dump(s, tag):
    for sd in s.all_sdfgs_recursive():
        for e in sd.all_interstate_edges():
            if e.data.assignments: print(tag, sd.name, 'ISE', e.data.assignments)
    for n, g in s.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry): print(tag, 'MAP', n.map.label, n.map.params, n.map.range)
        if isinstance(n, dace.nodes.NestedSDFG): print(tag, 'NSDFG', n.label, n.symbol_mapping)
        if isinstance(n, dace.nodes.Tasklet): print(tag, 'T', n.label, repr(n.code.as_string)[:150])
        if isinstance(n, dace.nodes.LibraryNode):
            print(tag, 'L', type(n).__name__, n.label, {k: getattr(n, k, None) for k in ('op','kind_a','kind_b','expr_a','expr_b','src_kind','dim_strides','gather_dims','src_dims')})
        if isinstance(n, dace.nodes.AccessNode):
            for e in g.out_edges(n): print(tag, '  E', n.data, '->', e.dst, e.dst_conn, e.data)
dump(s, 'CANON')
v = VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR, remainder_strategy='scalar_postamble', expand_tile_nodes=False))
v.apply_pass(s, {})
dump(s, 'VEC')
s.expand_library_nodes()
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet) and 'load' in n.label:
        print('EXP', n.label, [ (e.dst_conn, str(e.data)) for e in g.in_edges(n)], n.code.as_string)
