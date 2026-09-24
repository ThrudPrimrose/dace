# PYTHONHASHSEED=0 python repro_replicate_offset.py
import copy
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.utils.tile_access import classify_tile_access
from dace import subsets

N = dace.symbol('N')


@dace.program
def repl_off(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[(i + 1) // 2]


@dace.program
def repl_coef(a: dace.float64[2 * N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[(3 * i) // 2]


for sub in ("int_floor(i + 1, 2)", "int_floor(3*i, 2)"):
    r = classify_tile_access(subsets.Range.from_string(sub), iter_vars=("i", ))
    print(sub, "->", r.per_dim_kind, "dim_strides", r.dim_strides, "replicate", r.replicate_factor_per_dim)

n = 32
for prog, alen in ((repl_off, n), (repl_coef, 2 * n)):
    sdfg = prog.to_sdfg(simplify=True)
    ref = copy.deepcopy(sdfg)
    ref.name += "_ref"
    canonicalize(sdfg, validate=True)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa="SCALAR")).apply_pass(sdfg, {})
    sdfg.name += "_vec"
    a = np.arange(alen, dtype=np.float64)
    b_ref, b_vec = np.zeros(n), np.zeros(n)
    ref(a=a, b=b_ref, N=n)
    sdfg(a=a, b=b_vec, N=n)
    print(prog.name, "match:", np.array_equal(b_ref, b_vec))
    print("  ref", b_ref[:16])
    print("  vec", b_vec[:16])
