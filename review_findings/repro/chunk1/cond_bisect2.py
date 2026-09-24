import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize import pipeline as cp
from ivs_cond_increment import cond_counter, reference, run

a = np.array([0.9, 0.1, 0.7, 0.2, 0.8, 0.3])
rb, rk = reference(a)
stages = cp._build_stages()
lo, hi = 0, len(stages)  # prefix length lo is good, hi is bad
def ok(n):
    sdfg = cond_counter.to_sdfg(simplify=True)
    sdfg.name = f'cond_bis_{n}'
    with dace.symbolic.serialization_symbol_dtypes(dict(sdfg.symbols)):
        for label, unit in stages[:n]:
            unit.apply_pass(sdfg, {})
    gb, gk = run(sdfg, a)
    return np.array_equal(gb, rb) and gk == rk
while hi - lo > 1:
    mid = (lo + hi) // 2
    if ok(mid):
        lo = mid
    else:
        hi = mid
    print('lo', lo, 'hi', hi, flush=True)
print('first bad unit:', hi - 1, stages[hi - 1][0], type(stages[hi - 1][1]).__name__)
