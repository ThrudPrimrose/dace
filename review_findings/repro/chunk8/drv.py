import copy, dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim, _TILE_NODE_TYPES

def vec(sdfg, widths=(4,), **kw):
    VectorizeCPUMultiDim(VectorizeConfig(widths=widths, target_isa=kw.pop('isa', 'SCALAR'), **kw)).apply_pass(sdfg, {})
    return sdfg

def ntile(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, _TILE_NODE_TYPES))
