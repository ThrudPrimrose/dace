"""InductionVariableSubstitution closes ``for i in range(M, N): s += 2`` with trip count N - M,
which is negative when M > N (the loop runs zero times)."""
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.simplify import SimplifyPass
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution

M, N = (dace.symbol(s, dtype=dace.int64) for s in 'MN')


@dace.program
def counted(s: dace.float64[1], p: dace.float64[1]):
    for i in range(M, N):
        s[0] = s[0] + 2.0
    for j in range(M, N):
        p[0] = p[0] * 2.0


sdfg = counted.to_sdfg(simplify=True)
print('direct pass result:', InductionVariableSubstitution().apply_pass(sdfg, {}))
print('loops left:', sum(isinstance(b, LoopRegion) for b in sdfg.all_control_flow_blocks(recursive=True)))
s, p = np.array([10.0]), np.array([10.0])
sdfg(s=s, p=p, M=5, N=3)
print('direct:   expected s=10 p=10, got s =', s[0], 'p =', p[0])

sdfg = counted.to_sdfg(simplify=True)
sdfg.name = 'counted_pipeline'
canonicalize(sdfg, validate=True)
s, p = np.array([10.0]), np.array([10.0])
sdfg(s=s, p=p, M=5, N=3)
print('pipeline: expected s=10 p=10, got s =', s[0], 'p =', p[0])

# Isolate: after the 'clean' stage the loops are the single-tasklet shape IVS matches.
sdfg = counted.to_sdfg(simplify=True)
sdfg.name = 'counted_isolated'
canonicalize(sdfg, stages=['clean'])
promote = ScalarToSymbolPromotion()
promote.transients_only = False
promote.apply_pass(sdfg, {})
SimplifyPass().apply_pass(sdfg, {})
print('after clean + promote + simplify, IVS result:', InductionVariableSubstitution().apply_pass(sdfg, {}))
s, p = np.array([10.0]), np.array([10.0])
sdfg(s=s, p=p, M=5, N=3)
print('isolated: expected s=10 p=10, got s =', s[0], 'p =', p[0])
