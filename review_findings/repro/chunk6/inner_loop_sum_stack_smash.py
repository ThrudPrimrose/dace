"""Out of chunk6: a per-row sum over a sequential inner loop. The canonical Reduce(axes=[0]) reads a
1-D NSDFG-local view; _RunExpandNestedSDFGInputs rewrites its input to the 2-D ``A[i, 0:n]`` without
remapping ``axes``, so the Reduce expands to a copy of n elements into the scalar ``s``."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def inner_bound(cnt: dace.int32[1], A: dace.float64[M, N], out: dace.float64[M]):
    n = cnt[0]
    for i in dace.map[0:M]:
        s = 0.0
        for k in range(n):
            s = s + A[i, k]
        out[i] = s


m, nn = 5, 12
A = np.random.default_rng(0).random((m, nn))
out = np.zeros(m)
sdfg = inner_bound.to_sdfg(simplify=True)
sdfg.name = 'inner_bound_stack_smash'
canonicalize(sdfg, validate=True)
VectorizeCPUMultiDim(VectorizeConfig(widths=(8, ), target_isa=ISA.SCALAR, remainder_strategy='masked_tail')).apply_pass(sdfg, {})
sdfg(cnt=np.array([7], np.int32), A=A, out=out, N=nn, M=m)
print('max err', np.abs(out - A[:, :7].sum(axis=1)).max())
