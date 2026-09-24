"""Isolate LiftLoopCarriedReduction: canonicalize up to (not incl.) 'end', then apply the pass alone."""
import numpy as np
import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize, stage_labels
from dace.transformation.passes.canonicalize.lift_loop_carried_reduction import LiftLoopCarriedReduction
from llcr_prog import shared_partial

rng = np.random.default_rng(0)
X = rng.random((8, 4, 4))
P0 = rng.random((4, 4))
rP, rQ = P0.copy(), np.zeros((4, 4))
for idx in range(8):
    rQ[:] = rP + X[idx]
    rP[:] = rQ

labels = stage_labels()
for apply_lift in (False, True):
    sdfg = shared_partial.to_sdfg(simplify=True)
    sdfg.name = f'llcr_iso_{int(apply_lift)}'
    canonicalize(sdfg, stages=[l for l in labels if l != 'end'])
    if apply_lift:
        print('LiftLoopCarriedReduction result:', LiftLoopCarriedReduction().apply_pass(sdfg, {}))
        for n, st in sdfg.all_nodes_recursive():
            if isinstance(n, nodes.Tasklet):
                print('   tasklet', repr(n.code.as_string), [str(e.data) for e in st.out_edges(n)])
    P, Q = P0.copy(), np.zeros((4, 4))
    sdfg(P=P, Q=Q, X=X)
    print('lift applied' if apply_lift else 'no lift     ', 'P max err', np.abs(P - rP).max(), 'Q max err',
          np.abs(Q - rQ).max())
