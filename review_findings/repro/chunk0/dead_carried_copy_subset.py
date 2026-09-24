"""DeadCarriedStoreElimination reads ``edge.data.subset`` as the destination subset of a store,
but on an AccessNode->AccessNode copy whose memlet names the SOURCE (``c[i] -> a[i + 2]``) that
subset is ``c``'s.  The copy into ``a[i + 2]`` is misread as a store at offset 0 and taken as the
kill of the real store ``a[i + 1]``, which is live and gets dropped.
"""
import numpy as np
import dace
from dace.transformation.passes.canonicalize.dead_carried_store import DeadCarriedStoreElimination

N = dace.symbol('N')


@dace.program
def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N - 2):
        a[i + 1] = b[i] * 2.0
        a[i + 2] = c[i]


def reference(a, b, c):
    a = a.copy()
    for i in range(a.shape[0] - 2):
        a[i + 1] = b[i] * 2.0
        a[i + 2] = c[i]
    return a


if __name__ == '__main__':
    n = 8
    rng = np.random.default_rng(0)
    a0, b, c = rng.random(n), rng.random(n), rng.random(n)
    sdfg = prog.to_sdfg(simplify=True)
    for st in sdfg.states():
        for e in st.edges():
            if e.data.data:
                print('   ', st.label, e.src, '->', e.dst, 'memlet', e.data)
    print('pass returned', DeadCarriedStoreElimination().apply_pass(sdfg, {}))
    sdfg.validate()
    a = a0.copy()
    sdfg(a=a, b=b, c=c, N=n)
    ref = reference(a0, b, c)
    print('numpy:', np.round(ref, 3))
    print('dace :', np.round(a, 3))
    print('MISMATCH' if not np.allclose(a, ref) else 'ok')
