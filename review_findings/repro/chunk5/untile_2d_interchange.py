# UntileLoops' multi-dim ascent collapses (ti, tj, i, j) to (i, j): a loop interchange of tj and i
# done without any dependence check. A (1, -1) flow dependence crosses tile columns, so the
# tiled program order and the row-major order compute different values.
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops

N = 8

@dace.program
def k(a: dace.float64[N + 1, N + 1]):
    for ti in range(0, N, 4):
        for tj in range(0, N, 4):
            for i in range(ti, ti + 4):
                for j in range(tj, tj + 4):
                    a[i + 1, j] = a[i, j + 1] * 0.5 + a[i + 1, j]

sdfg = k.to_sdfg(simplify=True)
rng = np.random.default_rng(0)
a0 = rng.random((N + 1, N + 1))
ref = a0.copy(); k.f(ref)
print('UntileLoops returned', UntileLoops().apply_pass(sdfg, {}))
print([(r.loop_variable, r.init_statement.as_string, r.loop_condition.as_string)
       for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)])
a = a0.copy(); sdfg(a=a)
print('max abs diff vs program:', np.abs(a - ref).max(), 'match', np.allclose(a, ref))
