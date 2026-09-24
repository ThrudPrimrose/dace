"""DeadCarriedStoreElimination never compares the non-carried indices of the dead and killing store.

a[i + 1, 1] is never overwritten (the other store writes column 0), yet it is dropped as
"killed by a[i, 0] one iteration later".
"""
import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize.dead_carried_store import DeadCarriedStoreElimination

N = dace.symbol('N')


@dace.program
def prog(a: dace.float64[N, 2], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N - 1):
        a[i, 0] = b[i]
        a[i + 1, 1] = c[i]


def reference(a, b, c):
    a = a.copy()
    for i in range(a.shape[0] - 1):
        a[i, 0] = b[i]
        a[i + 1, 1] = c[i]
    return a


if __name__ == '__main__':
    n = 6
    rng = np.random.default_rng(0)
    a0, b, c = rng.random((n, 2)), rng.random(n), rng.random(n)
    ref = reference(a0, b, c)
    if len(sys.argv) > 1 and sys.argv[1] == 'canonicalize':
        from dace.transformation.passes.canonicalize import canonicalize
        sdfg = canonicalize(prog.to_sdfg())
    else:
        sdfg = prog.to_sdfg(simplify=True)
        print('pass returned', DeadCarriedStoreElimination().apply_pass(sdfg, {}))
        sdfg.validate()
    a = a0.copy()
    sdfg(a=a, b=b, c=c, N=n)
    print('max |diff| column 1:', np.max(np.abs(a[:, 1] - ref[:, 1])))
    print('MISMATCH' if not np.allclose(a, ref) else 'ok')
