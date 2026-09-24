# Case B: bare outer iterator `i` in tasklet code survives the rewrite while the loop that defined it is gone.
import numpy as np
import dace
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops

N = 16

@dace.program
def k(a: dace.float64[N]):
    for i in range(0, N, 4):
        for ii in range(i, i + 4):
            a[ii] = i

sdfg = k.to_sdfg(simplify=True)
print('UntileLoops returned', UntileLoops().apply_pass(sdfg, {}))
for st in sdfg.all_states():
    for n in st.nodes():
        if isinstance(n, dace.nodes.Tasklet):
            print('after tasklet:', n.code.as_string)
print('free symbols:', sorted(sdfg.free_symbols), 'arglist:', list(sdfg.arglist().keys()))
try:
    sdfg.validate(); print('validates')
except Exception as e:
    print('validate error:', type(e).__name__, str(e)[:200])
