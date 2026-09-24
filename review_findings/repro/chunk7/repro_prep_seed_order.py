"""PrepareReductionForWidening: in-state seed of the private accumulator is unordered w.r.t. the map."""
import copy
import numpy as np, dace
from drv import compare, run, vectorize
from dace.transformation.passes.vectorization.reduction_scalar_local_prep import PrepareReductionForWidening

N = dace.symbol('N')


@dace.program
def k(a: dace.float64[N], s: dace.float64[4], c: dace.float64[1]):
    s[3] = 5.0
    for i in dace.map[0:N]:
        s[3] += a[i]
    c[0] = s[3] * 2.0


n = 19
a = np.random.default_rng(0).random(n)
s = np.ones(4)
ref_s = s.copy()
ref_s[3] = 5.0 + a.sum()
ref = {"s": ref_s, "c": np.array([2 * ref_s[3]])}
base = k.to_sdfg(simplify=True)
plain = copy.deepcopy(base)
plain.name = "rp_seed_plain"
print("simplified, no prep:", end=" ")
compare(ref, run(plain, {"a": a, "s": s, "c": np.zeros(1)}, N=n))
prep = copy.deepcopy(base)
prep.name = "rp_seed_prep"
print("PrepareReductionForWidening returned", PrepareReductionForWidening().apply_pass(prep, {}))
prep.validate()
print("after prep:", end=" ")
compare(ref, run(prep, {"a": a, "s": s, "c": np.zeros(1)}, N=n))
vec = vectorize(None, "rp_seed_vec", sdfg=copy.deepcopy(base), canon=False)
print("VectorizeCPUMultiDim (uncanonicalized input):", end=" ")
compare(ref, run(vec, {"a": a, "s": s, "c": np.zeros(1)}, N=n))
