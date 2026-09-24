"""InductionVariableSubstitution closes ``for i in range(M, N): s += 2`` with trip count N - M,
which is negative when M > N (the loop runs zero times)."""
import numpy as np
import dace
from dace.sdfg.state import LoopRegion

M, N = (dace.symbol(s, dtype=dace.int64) for s in 'MN')


@dace.program
def counted(s: dace.float64[1], p: dace.float64[1]):
    for i in range(M, N):
        s[0] = s[0] + 2.0
    for j in range(M, N):
        p[0] = p[0] * 2.0


