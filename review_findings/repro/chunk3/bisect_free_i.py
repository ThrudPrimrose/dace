import dace
from dace.transformation.passes.canonicalize import CANONICALIZE_STAGES
N = dace.symbol('N')

@dace.program
def postloop(A: dace.float64[10, 10], B: dace.float64[10], C: dace.int64[1]):
    for i in range(N):
        for j in range(N):
            A[i, j] = A[i, j] + 1.0
        B[i] = B[i] + 2.0
    C[0] = i

sdfg = postloop.to_sdfg(simplify=True)
print('initial free', sorted(sdfg.free_symbols))
for label, factory in CANONICALIZE_STAGES:
    for unit in factory():
        unit.apply_pass(sdfg, {})
        if 'i' in sdfg.free_symbols:
            print('first introduces free i:', label, type(unit).__name__)
            raise SystemExit
print('never')
