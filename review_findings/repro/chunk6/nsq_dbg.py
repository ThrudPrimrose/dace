import copy, warnings
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
N = dace.symbol('N')

@dace.program
def k_nsq(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] - (N * N) / 7

s = k_nsq.to_sdfg(simplify=True)
canonicalize(s, validate=True)
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet): print('CANON', n.label, repr(n.code.as_string), dict(n.in_connectors), dict(n.out_connectors))
for sd in s.all_sdfgs_recursive():
    print(sd.name, dict(sd.symbols), {k: (type(v).__name__, v.dtype, getattr(v, 'shape', None)) for k, v in sd.arrays.items()})
VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR, remainder_strategy='masked_tail', expand_tile_nodes=False)).apply_pass(s, {})
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet): print('T', n.label, repr(n.code.as_string)[:200])
    elif isinstance(n, dace.nodes.LibraryNode): print('L', type(n).__name__, n.label, {k: getattr(n, k, None) for k in ('op','kind_a','kind_b','expr_a','expr_b','src_kind','src_expr')})
for sd in s.all_sdfgs_recursive():
    print(sd.name, {k: (type(v).__name__, v.dtype, getattr(v, 'shape', None)) for k, v in sd.arrays.items() })
s.expand_library_nodes()
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet) and 'split' in n.label: print('EXP', n.label, dict(n.in_connectors), dict(n.out_connectors), n.code.as_string[:600])
