"""LoopCarriedRotationSubstitution peels one unguarded iteration off a loop whose trip count is symbolic."""
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.induction_variable_substitution import LoopCarriedRotationSubstitution

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def delay_line(a: dace.float64[10], b: dace.float64[10], c: dace.float64[1]):
    x = c[0]
    for i in range(N):
        a[i] = (b[i] + x) * 0.5
        x = b[i]


def reference(a, b, c, n):
    a = a.copy()
    x = c[0]
    for i in range(n):
        a[i] = (b[i] + x) * 0.5
        x = b[i]
    return a


def run(sdfg, n):
    a, b, c = np.full(10, -1.0), np.arange(10.0), np.array([7.0])
    sdfg(a=a, b=b, c=c, N=n)
    return a, reference(np.full(10, -1.0), b, c, n)


if __name__ == '__main__':
    sdfg = delay_line.to_sdfg(simplify=True)
    sdfg.name = 'delay_line_canon'
    canonicalize(sdfg, validate=True)
    for n in (0, 4):
        got, ref = run(sdfg, n)
        print(f'N={n} canonicalized a[:5] =', got[:5], ' expected', ref[:5])
