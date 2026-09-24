# UntileLoops: a branch on the tile-local index `ii` (first element of each tile) is not audited;
# case A substitutes ii -> k - i, i -> 0, so `ii == 0` becomes `k == 0` (first element overall).
import numpy as np
import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops

N = 16

@dace.program
def k(a: dace.float64[N]):
    for i in range(0, N, 4):
        for ii in range(0, 4):
            if ii == 0:
                a[i + ii] = 1.0

sdfg = k.to_sdfg(simplify=True)
ref = np.zeros(N); k.f(ref)
print('UntileLoops returned', UntileLoops().apply_pass(sdfg, {}))
print('conditions after:', [c.as_string for b in sdfg.all_control_flow_blocks() if isinstance(b, ConditionalBlock)
                            for c, _ in b.branches if c is not None])
a = np.zeros(N); sdfg(a=a)
print('ref', ref); print('got', a); print('match', np.array_equal(a, ref))
