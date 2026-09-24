# PYTHONHASHSEED=0 python repro_multi_elem_buffer_crash.py
import copy, json
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')


@dace.program
def k_buf(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        tmp = np.empty(2, dtype=np.float64)
        tmp[0] = a[i]
        tmp[1] = a[i] * 2.0
        b[i] = tmp[0] + tmp[1]


sdfg = k_buf.to_sdfg(simplify=True)
canonicalize(sdfg, validate=True)
before = json.dumps(sdfg.to_json(), sort_keys=True)
try:
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa="SCALAR")).apply_pass(sdfg, {})
    print("no exception")
except Exception as ex:
    print("EXCEPTION:", type(ex).__name__, str(ex)[:160])
after = json.dumps(sdfg.to_json(), sort_keys=True)
print("SDFG unchanged after failure:", before == after)
try:
    sdfg.validate()
    print("SDFG still valid")
except Exception as ex:
    print("SDFG INVALID after failure:", type(ex).__name__, str(ex)[:160])
