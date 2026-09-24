# PYTHONHASHSEED=0 python repro_tasklet_index_constant.py
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization import widen_accesses as wa
from dace.transformation.passes.vectorization.utils.tile_access import classify_tile_access

N = dace.symbol('N')


@dace.program
def computed_gather(a: dace.float64[2 * N], idx: dace.int32[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        k = idx[i] * 2
        b[i] = a[k] + a[k + 1]


# Observe how WidenAccesses' classifier sees the ``a`` reads (read-only hook, delegates unchanged).
orig = wa.WidenAccesses.apply_pass


def observe(self, sdfg, pipeline_results):
    for _st, nsdfg, me, _w in self._body_nsdfgs(sdfg):
        inner = nsdfg.sdfg
        print("interstate:", [e.data.assignments for e in inner.all_interstate_edges() if e.data.assignments])
        for s in inner.states():
            for e in s.edges():
                if e.data.data == 'a':
                    r = classify_tile_access(e.data.subset, iter_vars=(me.map.params[-1], ), inner_sdfg=inner, state=s)
                    print(f"  a[{e.data.subset}] classified {[k.name for k in r.per_dim_kind]}")
        break
    return orig(self, sdfg, pipeline_results)


wa.WidenAccesses.apply_pass = observe
sdfg = computed_gather.to_sdfg(simplify=True)
canonicalize(sdfg, validate=True)
VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa="SCALAR")).apply_pass(sdfg, {})
sdfg.validate()
print("validate: OK")
n = 29
try:
    sdfg(a=np.random.rand(2 * n), idx=np.arange(n, dtype=np.int32)[::-1].copy(), b=np.zeros(n), N=n)
    print("ran")
except Exception as ex:
    print("COMPILE/RUN FAILURE:", type(ex).__name__, [l.strip() for l in str(ex).splitlines() if "error" in l][:1])
