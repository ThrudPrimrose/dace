import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.libraries.tileops import TileMaskGen
N = dace.symbol('N')
@dace.program
def flat_square(A: dace.float64[N * N], C: dace.float64[N * N]):
    for i in dace.map[0:N * N]:
        C[i] = A[i] * 2.0
s = flat_square.to_sdfg(simplify=True)
canonicalize(s, validate=True)
VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR, remainder_strategy='full_mask', expand_tile_nodes=False)).apply_pass(s, {})
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.MapEntry): print('MAP', n.map.label, n.map.params, n.map.range)
    if isinstance(n, TileMaskGen): print('MASK', n.iter_vars, n.global_ubs)
s.expand_library_nodes()
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet) and 'mask' in n.label: print(n.code.as_string)
