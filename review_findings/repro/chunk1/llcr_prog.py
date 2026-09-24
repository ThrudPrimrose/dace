"""LiftLoopCarriedReduction: the reduction's partial result also feeds a second output."""
import numpy as np
import dace


@dace.program
def shared_partial(P: dace.float64[4, 4], Q: dace.float64[4, 4], X: dace.float64[8, 4, 4]):
    for idx in range(8):
        for i, j in dace.map[0:4, 0:4]:
            t = P[i, j] + X[idx, i, j]
            P[i, j] = t
            Q[i, j] = t


