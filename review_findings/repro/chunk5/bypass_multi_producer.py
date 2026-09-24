# BypassTrivialAssignTasklets (src branch) re-points EVERY producer of the transient source onto the
# copy's destination subset, without checking that the producer wrote the element the copy reads.
#   T[0] = 1; T[1] = 2; out[0] = T[0]   ->   out[0] = 1; out[0] = 2
import numpy as np
import dace
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import BypassTrivialAssignTasklets


def build():
    inner = dace.SDFG('inner')
    inner.add_array('out', [1], dace.float64)
    inner.add_array('T', [2], dace.float64, transient=True)
    st = inner.add_state()
    t_an = st.add_access('T')
    p1 = st.add_tasklet('p1', {}, {'_out'}, '_out = 1.0')
    p2 = st.add_tasklet('p2', {}, {'_out'}, '_out = 2.0')
    st.add_edge(p1, '_out', t_an, None, dace.Memlet('T[0]'))
    st.add_edge(p2, '_out', t_an, None, dace.Memlet('T[1]'))
    cp = st.add_tasklet('copy', {'_in'}, {'_out'}, '_out = _in')
    st.add_edge(t_an, None, cp, '_in', dace.Memlet('T[0]'))
    st.add_edge(cp, '_out', st.add_write('out'), None, dace.Memlet('out[0]'))

    outer = dace.SDFG('bypass_multi_producer')
    outer.add_array('out', [1], dace.float64)
    ost = outer.add_state()
    ns = ost.add_nested_sdfg(inner, {}, {'out'})
    ost.add_edge(ns, 'out', ost.add_write('out'), None, dace.Memlet('out[0]'))
    return outer


ref = np.zeros(1); build()(out=ref)
sdfg = build()
print('pass returned', BypassTrivialAssignTasklets().apply_pass(sdfg, {}))
inner = sdfg.start_state.nodes()[0].sdfg if isinstance(sdfg.start_state.nodes()[0], dace.nodes.NestedSDFG) else \
    [n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)][0].sdfg
for e in inner.start_state.edges():
    print('  inner edge:', e.src, '->', e.dst, e.data)
sdfg.validate()
got = np.zeros(1); sdfg(out=got)
print('ref', ref, 'got', got, 'match', np.array_equal(ref, got))
