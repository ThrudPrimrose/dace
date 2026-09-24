# UntileLoops audits only memlet subsets; a bare `i` in tasklet code gets substituted by 0.
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops

N = 16

@dace.program
def k(a: dace.float64[N]):
    for i in range(0, N, 4):
        for ii in range(0, 4):
            a[i + ii] = i

sdfg = k.to_sdfg(simplify=True)
for st in sdfg.all_states():
    for n in st.nodes():
        if isinstance(n, dace.nodes.Tasklet):
            print('before tasklet:', n.code.as_string)
ref = np.zeros(N); k.f(ref)
res = UntileLoops().apply_pass(sdfg, {})
print('UntileLoops returned', res)
for st in sdfg.all_states():
    for n in st.nodes():
        if isinstance(n, dace.nodes.Tasklet):
            print('after tasklet:', n.code.as_string)
print([ (r.loop_variable, r.init_statement.as_string, r.loop_condition.as_string) for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion)])
a = np.zeros(N); sdfg(a=a)
print('ref', ref); print('got', a); print('match', np.allclose(a, ref))
