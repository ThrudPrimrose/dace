import numpy as np
import dace
from dace.transformation.passes.canonicalize.perfect_loop_nesting import PerfectLoopNesting

N = dace.symbol('N')

@dace.program
def prog(A: dace.float64[10, 10], B: dace.float64[10], C: dace.int64[1]):
    for i in range(N):
        for j in range(N):
            A[i, j] = A[i, j] + 1.0
        B[i] = B[i] + 2.0
    C[0] = i

def run(sdfg):
    A = np.zeros((10, 10)); B = np.zeros(10); C = np.zeros(1, dtype=np.int64)
    sdfg(A=A, B=B, C=C, N=5)
    return C[0]

sdfg = prog.to_sdfg(simplify=True)
print('before pass: C =', run(sdfg))
sdfg = prog.to_sdfg(simplify=True)
print('pass returned', PerfectLoopNesting().apply_pass(sdfg, {}))
print([ (r.label, r.loop_variable) for r in sdfg.all_control_flow_regions() if isinstance(r, dace.sdfg.state.LoopRegion)])
print('free symbols:', sdfg.free_symbols)
sdfg.validate()
sdfg.name = 'prog_after'
print('after pass: C =', run(sdfg))
