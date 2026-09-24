"""LiftLoopCarriedReduction: the reduction's partial result also feeds a second output."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize


@dace.program
def shared_partial(P: dace.float64[4, 4], Q: dace.float64[4, 4], X: dace.float64[8, 4, 4]):
    for idx in range(8):
        for i, j in dace.map[0:4, 0:4]:
            t = P[i, j] + X[idx, i, j]
            P[i, j] = t
            Q[i, j] = t


rng = np.random.default_rng(0)
X = rng.random((8, 4, 4))
P0 = rng.random((4, 4))
rP, rQ = P0.copy(), np.zeros((4, 4))
for idx in range(8):
    rQ[:] = rP + X[idx]
    rP[:] = rQ
sdfg = shared_partial.to_sdfg(simplify=True)
canonicalize(sdfg, validate=True)
P, Q = P0.copy(), np.zeros((4, 4))
sdfg(P=P, Q=Q, X=X)
print('P max err', np.abs(P - rP).max(), 'Q max err', np.abs(Q - rQ).max())
