import sys, copy
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.demote_data_reading_interstate_symbols import DemoteDataReadingInterstateSymbols
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def inner_bound(cnt: dace.int32[1], A: dace.float64[M, N], out: dace.float64[M]):
    n = cnt[0]
    for i in dace.map[0:M]:
        s = 0.0
        for k in range(n):
            s = s + A[i, k]
        out[i] = s

s = inner_bound.to_sdfg(simplify=True)
canonicalize(s, validate=True)
s.save('ib_canon.sdfg')
VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR, remainder_strategy='masked_tail')).apply_pass(s, {})
s.save('ib_vec.sdfg')
for sd in s.all_sdfgs_recursive():
    print(sd.name, {k: (type(v).__name__, str(v.dtype), v.shape, str(v.storage), v.transient) for k, v in sd.arrays.items()})
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.MapEntry): print('MAP', n.map.label, n.map.range, n.map.schedule)
    if isinstance(n, dace.nodes.Tasklet): print('T', n.label, n.code.as_string[:200], [(e.dst_conn, str(e.data)) for e in g.in_edges(n)], [(e.src_conn, str(e.data)) for e in g.out_edges(n)])
    if isinstance(n, dace.nodes.NestedSDFG): print('NS', n.label, n.symbol_mapping, [(e.dst_conn, str(e.data)) for e in g.in_edges(n)], [(e.src_conn, str(e.data)) for e in g.out_edges(n)])
for sd in s.all_sdfgs_recursive():
    for cfr in sd.all_control_flow_regions():
        if type(cfr).__name__ == 'LoopRegion': print('LOOP', sd.name, cfr.loop_variable, cfr.init_statement.as_string, cfr.loop_condition.as_string, cfr.update_statement.as_string)
    for e in sd.all_interstate_edges():
        if e.data.assignments: print('ISE', sd.name, e.data.assignments)
