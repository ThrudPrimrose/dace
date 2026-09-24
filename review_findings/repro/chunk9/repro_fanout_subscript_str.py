# PYTHONHASHSEED=0 python repro_fanout_subscript_str.py
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


@dace.program
def gather_plus_one(a: dace.float64[N + 1], idx: dace.int32[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        b[i] = a[idx[i] + 1]


sdfg = gather_plus_one.to_sdfg(simplify=True)
canonicalize(sdfg, validate=True)
VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa="SCALAR")).apply_pass(sdfg, {})
sdfg.validate()
print("validate: OK")
for e in sdfg.all_interstate_edges(recursive=True):
    for k, v in e.data.assignments.items():
        if "lane" in k:
            print("per-lane assignment:", k, "=", v)
n = 29
try:
    sdfg(a=np.random.rand(n + 1), idx=np.arange(n, dtype=np.int32), b=np.zeros(n), N=n)
    print("ran")
except Exception as ex:
    print("COMPILE/RUN FAILURE:", type(ex).__name__, [l for l in str(ex).splitlines() if "error" in l][:2])
