import sys, copy, warnings
sys.path.insert(0, '.')
import numpy as np, dace
from swapbuild import build
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
s = build(); s.name = 'swap_alias_vec'
if sys.argv[1] == 'canon':
    canonicalize(s, validate=True)
    ns = [x for x, _ in s.all_nodes_recursive() if isinstance(x, dace.nodes.NestedSDFG)]
    print("after canonicalize nsdfgs:", [dict(x.symbol_mapping) for x in ns])
try:
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa="SCALAR")).apply_pass(s, {})
    print("vectorize ok")
except Exception as ex:
    print("EXC", type(ex).__name__, str(ex)[:200])
