# BypassTrivialAssignTasklets (dst branch) rebuilds the consumer copy as
#   Memlet(data=src, subset=in_subset, other_subset=de.data.subset)
# but de.data.subset is the subset of de.data.data -- here the bypassed transient T, not the consumer C.
import numpy as np
import dace
from dace.transformation.passes.vectorization.bypass_trivial_assign_tasklets import BypassTrivialAssignTasklets


def build():
    inner = dace.SDFG('inner')
    inner.add_array('A', [10], dace.float64)
    inner.add_array('C', [10], dace.float64)
    inner.add_array('T', [2], dace.float64, transient=True)
    st = inner.add_state()
    a = st.add_read('A')
    t = st.add_access('T')
    c = st.add_write('C')
    tk = st.add_tasklet('assign', {'_in'}, {'_out'}, '_out = _in')
    st.add_edge(a, None, tk, '_in', dace.Memlet('A[5]'))
    st.add_edge(tk, '_out', t, None, dace.Memlet('T[1]'))
    # Standard AN->AN copy memlet: data names the SOURCE (T), other_subset is C's subset.
    st.add_edge(t, None, c, None, dace.Memlet('T[1] -> [3]'))

    outer = dace.SDFG('bypass_other_subset')
    outer.add_array('A', [10], dace.float64)
    outer.add_array('C', [10], dace.float64)
    ost = outer.add_state()
    ns = ost.add_nested_sdfg(inner, {'A'}, {'C'})
    ost.add_edge(ost.add_read('A'), None, ns, 'A', dace.Memlet('A[0:10]'))
    ost.add_edge(ns, 'C', ost.add_write('C'), None, dace.Memlet('C[0:10]'))
    return outer


A = np.arange(10, dtype=np.float64) + 100
ref = np.zeros(10); build()(A=A.copy(), C=ref)
sdfg = build()
print('pass returned', BypassTrivialAssignTasklets().apply_pass(sdfg, {}))
for st in sdfg.all_states():
    for e in st.edges():
        if st.sdfg is not sdfg:
            print('inner edge:', e.src, '->', e.dst, e.data)
sdfg.validate()
got = np.zeros(10); sdfg(A=A.copy(), C=got)
print('ref', ref); print('got', got); print('match', np.array_equal(ref, got))
