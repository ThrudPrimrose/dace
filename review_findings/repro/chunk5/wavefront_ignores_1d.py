# WavefrontSkew collects dependences only from 2-D arrays (scan_state_accesses skips every
# descriptor whose rank != 2), then lifts the tile column with parallelize_loop(proven=True).
# A 1-element accumulator carried through the nest is invisible to the legality test.
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.wavefront_skew import WavefrontSkew

N = 12

@dace.program
def k(a: dace.float64[N, N], s: dace.float64[1]):
    for i in range(1, N):
        for j in range(1, N):
            a[i, j] = 0.5 * a[i - 1, j] + 0.25 * a[i, j - 1] + 0.001 * s[0]
            s[0] = 0.5 * s[0] + a[i, j]

sdfg = k.to_sdfg(simplify=True)
rng = np.random.default_rng(0)
a0 = rng.random((N, N))
ra, rs = a0.copy(), np.zeros(1); k.f(ra, rs)
ws = WavefrontSkew()
ws.tile_i = ws.tile_j = 4
print('WavefrontSkew returned', ws.apply_pass(sdfg, {}))
print([(r.loop_variable, r.pinned_sequential) for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)])
sdfg.validate()
ga, gs = a0.copy(), np.zeros(1)
sdfg(a=ga, s=gs)
print('s ref', rs, 'got', gs)
print('a max abs diff', np.abs(ga - ra).max(), 'match', np.allclose(ga, ra) and np.allclose(gs, rs))
