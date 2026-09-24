"""Full canonicalize of dead_carried_copy_subset.prog, run at N = 2 (zero-trip loop)."""
import numpy as np
from dace.transformation.passes.canonicalize import dead_carried_store as dcs
dcs.DeadCarriedStoreElimination.apply_pass = lambda self, sdfg, res: None
from dace.transformation.passes.canonicalize import canonicalize
from dead_carried_copy_subset import prog, reference

sdfg = canonicalize(prog.to_sdfg())
csdfg = sdfg.compile()
for n in (2, 3, 5):
    rng = np.random.default_rng(0)
    a0, b, c = rng.random(n + 4), rng.random(n + 4), rng.random(n + 4)
    a = a0.copy()
    csdfg(a=a, b=b, c=c, N=n)
    ref = a0.copy()
    ref[:n] = reference(a0[:n], b[:n], c[:n])
    print(n, 'ok' if np.allclose(a, ref) else f'MISMATCH dace={np.round(a, 3)} ref={np.round(ref, 3)}')
