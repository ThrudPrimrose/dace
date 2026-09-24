"""Full canonicalize of dead_carried_other_axis.prog with DeadCarriedStoreElimination disabled."""
import numpy as np
from dace.transformation.passes.canonicalize import dead_carried_store as dcs
dcs.DeadCarriedStoreElimination.apply_pass = lambda self, sdfg, res: None
from dace.transformation.passes.canonicalize import canonicalize
from dead_carried_other_axis import prog, reference

n = 6
rng = np.random.default_rng(0)
a0, b, c = rng.random((n, 2)), rng.random(n), rng.random(n)
sdfg = canonicalize(prog.to_sdfg())
a = a0.copy()
sdfg(a=a, b=b, c=c, N=n)
print('with pass disabled:', 'ok' if np.allclose(a, reference(a0, b, c)) else 'MISMATCH')
