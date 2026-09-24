import warnings
import numpy as np, dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeMultiDim, _TILE_NODE_TYPES
N = dace.symbol('N')
@dace.program
def add1(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[i] + 1.0
sdfg = add1.to_sdfg(simplify=True); canonicalize(sdfg, validate=True)
p = VectorizeMultiDim(VectorizeConfig(widths=(8,), target_isa="CUDA_WARP"))
print("device:", p._device)
p.apply_pass(sdfg, {})
for n, st in sdfg.all_nodes_recursive():
    if isinstance(n, _TILE_NODE_TYPES):
        print(type(n).__name__, n.implementation, "schedule of enclosing map:", st.entry_node(n) and st.entry_node(n).map.schedule, "gpu storage?", {sdfg.arrays[k].storage.name for k in ('a','b')})
        break
