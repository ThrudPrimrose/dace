# Mirror of bypass_other_subset.py for the src branch: producer edge A -> T carries a memlet that
# names its DESTINATION (data='T', subset=T's index, other_subset=A's index). The bypass relabels
# pe.data.subset (T's index) as A's subset.
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
    st.add_edge(a, None, t, None, dace.Memlet(data='T', subset='1', other_subset='5'))  # A[5] -> T[1]
    tk = st.add_tasklet('assign', {'_in'}, {'_out'}, '_out = _in')
    st.add_edge(t, None, tk, '_in', dace.Memlet('T[1]'))
    st.add_edge(tk, '_out', st.add_write('C'), None, dace.Memlet('C[3]'))

    outer = dace.SDFG('bypass_other_subset_src')
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
sdfg.validate()
got = np.zeros(10); sdfg(A=A.copy(), C=got)
print('ref', ref); print('got', got); print('match', np.array_equal(ref, got))
