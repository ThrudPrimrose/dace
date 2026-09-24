import sys, copy
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.demote_data_reading_interstate_symbols import DemoteDataReadingInterstateSymbols
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

mode = sys.argv[1]
m, nn = 5, 12
rng = np.random.default_rng(0)
A = rng.random((m, nn))
out = np.zeros(m)
s = inner_bound.to_sdfg(simplify=True)
s.name = 'inner_bound_' + mode
if mode in ('canon', 'demote', 'vec'):
    canonicalize(s, validate=True)
if mode == 'demote':
    print('demoted', DemoteDataReadingInterstateSymbols().apply_pass(s, {}))
if mode == 'vec':
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR, remainder_strategy='masked_tail')).apply_pass(s, {})
s.validate()
s(cnt=np.array([7], np.int32), A=A, out=out, N=nn, M=m)
print(mode, 'max err', np.abs(out - A[:, :7].sum(axis=1)).max())
