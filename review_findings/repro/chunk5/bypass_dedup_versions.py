# BypassTrivialAssignTasklets._dedup_identity_assigns keys copies by (src name, subset, dst name, subset)
# only, so two copies of x[0] taken BEFORE and AFTER x[0] is overwritten are "deduplicated": the
# second copy's consumer is rewired onto the first copy's (stale) value.
#   T = x[0]; x[0] = T + 1; T = x[0]; out[0] = 2 * T
import numpy as np
import dace
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import BypassTrivialAssignTasklets


def build():
    inner = dace.SDFG('inner')
    inner.add_array('x', [1], dace.float64)
    inner.add_array('out', [1], dace.float64)
    inner.add_array('T', [1], dace.float64, transient=True)
    inner.add_array('old', [1], dace.float64)
    st = inner.add_state()
    x_a = st.add_read('x')
    t1 = st.add_tasklet('copy1', {'_in'}, {'_out'}, '_out = _in')
    T1 = st.add_access('T')
    w = st.add_tasklet('inc', {'_in'}, {'_out'}, '_out = _in + 1')
    x_b = st.add_access('x')
    t2 = st.add_tasklet('copy2', {'_in'}, {'_out'}, '_out = _in')
    T2 = st.add_access('T')
    dbl = st.add_tasklet('dbl', {'_in'}, {'_out'}, '_out = 2 * _in')
    o = st.add_write('out')
    st.add_edge(x_a, None, t1, '_in', dace.Memlet('x[0]'))
    st.add_edge(t1, '_out', T1, None, dace.Memlet('T[0]'))
    st.add_edge(T1, None, w, '_in', dace.Memlet('T[0]'))
    st.add_edge(w, '_out', x_b, None, dace.Memlet('x[0]'))
    st.add_edge(x_b, None, t2, '_in', dace.Memlet('x[0]'))
    st.add_edge(t2, '_out', T2, None, dace.Memlet('T[0]'))
    st.add_edge(T2, None, dbl, '_in', dace.Memlet('T[0]'))
    st.add_edge(dbl, '_out', o, None, dace.Memlet('out[0]'))
    # A second reader of the original x[0] (keeps x_a's out-degree at 2, so the later bypass step
    # leaves the stale T in place and the wrong value is deterministic rather than a race).
    neg = st.add_tasklet('neg', {'_in'}, {'_out'}, '_out = -_in')
    st.add_edge(x_a, None, neg, '_in', dace.Memlet('x[0]'))
    st.add_edge(neg, '_out', st.add_write('old'), None, dace.Memlet('old[0]'))

    outer = dace.SDFG('bypass_dedup_versions')
    outer.add_array('x', [1], dace.float64)
    outer.add_array('out', [1], dace.float64)
    outer.add_array('old', [1], dace.float64)
    ost = outer.add_state()
    ns = ost.add_nested_sdfg(inner, {'x'}, {'x', 'out', 'old'})
    ost.add_edge(ost.add_read('x'), None, ns, 'x', dace.Memlet('x[0]'))
    ost.add_edge(ns, 'x', ost.add_write('x'), None, dace.Memlet('x[0]'))
    ost.add_edge(ns, 'out', ost.add_write('out'), None, dace.Memlet('out[0]'))
    ost.add_edge(ns, 'old', ost.add_write('old'), None, dace.Memlet('old[0]'))
    return outer


ref_x, ref_o = np.array([10.0]), np.zeros(1)
build()(x=ref_x, out=ref_o, old=np.zeros(1))
sdfg = build()
print('pass returned', BypassTrivialAssignTasklets().apply_pass(sdfg, {}))
sdfg.validate()
x, o = np.array([10.0]), np.zeros(1)
sdfg(x=x, out=o, old=np.zeros(1))
print('ref x, out =', ref_x, ref_o)
print('got x, out =', x, o)
print('match', np.array_equal(ref_o, o) and np.array_equal(ref_x, x))
