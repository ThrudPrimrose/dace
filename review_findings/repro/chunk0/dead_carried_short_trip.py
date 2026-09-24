"""DeadCarriedStoreElimination peels `distance` tail iterations: does the peel stay guarded when
the loop runs fewer iterations than it peels (here zero)?"""
import numpy as np
import dace
from dace.transformation.passes.canonicalize.dead_carried_store import DeadCarriedStoreElimination
from dead_carried_copy_subset import prog, reference

if __name__ == '__main__':
    sdfg = prog.to_sdfg(simplify=True)
    print('pass returned', DeadCarriedStoreElimination().apply_pass(sdfg, {}))
    csdfg = sdfg.compile()
    for n in (2, 3, 4):
        rng = np.random.default_rng(0)
        a0, b, c = rng.random(n + 4), rng.random(n + 4), rng.random(n + 4)
        a = a0.copy()
        csdfg(a=a, b=b, c=c, N=n)
        ref = a0.copy()
        ref[:n] = reference(a0[:n], b[:n], c[:n])
        print(n, 'ok' if np.allclose(a, ref) else f'MISMATCH dace={np.round(a, 3)} ref={np.round(ref, 3)}')
