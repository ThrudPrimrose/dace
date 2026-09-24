"""Variant: live store a[i + 1] (tasklet) and a copy c[i] -> a[i + 2] after EliminateTrivialTasklets
turned the copy tasklet into an AccessNode->AccessNode edge.  Real kill relation: none for a[i+1]
(a[i+1] is written by the copy one iteration EARLIER), so nothing may be dropped."""
import numpy as np
import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize.dead_carried_store import DeadCarriedStoreElimination
from dace.transformation.passes.canonicalize.eliminate_trivial_tasklets import EliminateTrivialTasklets

N = dace.symbol('N')


@dace.program
def prog(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in range(N - 2):
        a[i + 2] = c[i]
        a[i + 1] = b[i] * 2.0


def reference(a, b, c):
    a = a.copy()
    for i in range(a.shape[0] - 2):
        a[i + 2] = c[i]
        a[i + 1] = b[i] * 2.0
    return a


if __name__ == '__main__':
    n = 8
    rng = np.random.default_rng(0)
    a0, b, c = rng.random(n), rng.random(n), rng.random(n)
    sdfg = prog.to_sdfg(simplify=True)
    print('trivial tasklets removed', EliminateTrivialTasklets().apply_pass(sdfg, {}))
    for st in sdfg.states():
        for e in st.edges():
            if isinstance(e.dst, nodes.AccessNode) and e.dst.data == 'a':
                print('   write into a:', e.src, '->', e.dst, 'memlet', e.data)
    print('pass returned', DeadCarriedStoreElimination().apply_pass(sdfg, {}))
    sdfg.validate()
    a = a0.copy()
    sdfg(a=a, b=b, c=c, N=n)
    ref = reference(a0, b, c)
    print('numpy:', np.round(ref, 3))
    print('dace :', np.round(a, 3))
    print('MISMATCH' if not np.allclose(a, ref) else 'ok')
