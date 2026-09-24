"""Bisect the canonicalize stage list for the first stage that breaks distribute_frontend.prog."""
import copy
import sys
import numpy as np
from dace import symbolic
from dace.transformation.passes.canonicalize.pipeline import _build_stages
from distribute_frontend import prog, reference
from dace.transformation.passes.canonicalize import distribute_producer_consumer as dpc
dpc.DistributeProducerConsumerLoop.apply_pass = lambda self, sdfg, res: None

n = 6
rng = np.random.default_rng(0)
a0, b = rng.random(n), rng.random(n)
ra, rt, ru = a0.copy(), np.zeros(n), np.zeros(n)
reference(ra, b, rt, ru)
base = prog.to_sdfg()
authority = {k: v for s in base.all_sdfgs_recursive() for k, v in s.symbols.items()}
stages = _build_stages()
cnt = [0]


def ok_after(k):
    with symbolic.serialization_symbol_dtypes(authority):
        sdfg = copy.deepcopy(base)
        for label, unit in stages[:k]:
            unit.apply_pass(sdfg, {})
    cnt[0] += 1
    sdfg.name = f'bis{cnt[0]}'
    a, t, u = a0.copy(), np.zeros(n), np.zeros(n)
    sdfg(a=a, b=b, t=t, u=u, N=n)
    return np.allclose(a, ra) and np.allclose(t, rt) and np.allclose(u, ru)


lo, hi = 0, len(stages)
print('total', hi, 'full ok?', ok_after(hi))
while hi - lo > 1:
    mid = (lo + hi) // 2
    if ok_after(mid):
        lo = mid
    else:
        hi = mid
    print(lo, hi, flush=True)
print('first bad stage index', hi - 1, stages[hi - 1][0], type(stages[hi - 1][1]).__name__)
