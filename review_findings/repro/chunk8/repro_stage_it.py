"""StageGlobalArrayThroughScalars drops the value of B[i] when a later write to B[i] is a conditional (IT) write."""
import copy, sys
import numpy as np
import dace
from drv import vec, ntile

N = 16


def build() -> dace.SDFG:
    sdfg = dace.SDFG('stage_then_masked_write')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('M', [N], dace.bool_)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('C', [N], dace.float64)
    st = sdfg.add_state()
    me, mx = st.add_map('m', dict(i=f'0:{N}'))
    a, m = st.add_read('A'), st.add_read('M')
    b1, b2, c = st.add_access('B'), st.add_access('B'), st.add_write('C')
    t0 = st.add_tasklet('t0', {'__a'}, {'__out'}, '__out = __a + 1.0')
    t1 = st.add_tasklet('t1', {'__b'}, {'__out'}, '__out = __b * 2.0')
    t2 = st.add_tasklet('t2', {'__in_cond'}, {'__out'}, 'if __in_cond:\n    __out = 7.0')
    st.add_memlet_path(a, me, t0, dst_conn='__a', memlet=dace.Memlet('A[i]'))
    st.add_memlet_path(m, me, t2, dst_conn='__in_cond', memlet=dace.Memlet('M[i]'))
    st.add_edge(t0, '__out', b1, None, dace.Memlet('B[i]'))
    st.add_edge(b1, None, t1, '__b', dace.Memlet('B[i]'))
    st.add_edge(t1, None, t2, None, dace.Memlet())  # program order: t2 after t1
    st.add_memlet_path(t1, mx, c, src_conn='__out', memlet=dace.Memlet('C[i]'))
    st.add_edge(t2, '__out', b2, None, dace.Memlet('B[i]'))
    st.add_memlet_path(b2, mx, st.add_write('B'), memlet=dace.Memlet('B[i]'))
    st.add_edge(b1, None, mx, None, dace.Memlet())  # keep b1 inside the scope
    sdfg.validate()
    return sdfg


def run(sdfg: dace.SDFG):
    rng = np.random.default_rng(0)
    A = rng.random(N)
    M = A > 0.5
    B = np.full(N, -1.0)
    C = np.zeros(N)
    sdfg(A=A, M=M, B=B, C=C)
    return B, C, np.where(M, 7.0, A + 1.0), 2 * (A + 1.0)


if __name__ == '__main__':
    ref_sdfg = build()
    B, C, Bref, Cref = run(ref_sdfg)
    print('unvectorized: B ok', np.allclose(B, Bref), 'C ok', np.allclose(C, Cref))
    sdfg = build()
    vec(sdfg)
    print('tile ops', ntile(sdfg))
    B, C, Bref, Cref = run(sdfg)
    print('vectorized:   B ok', np.allclose(B, Bref), 'C ok', np.allclose(C, Cref))
    print('B   ', B)
    print('Bref', Bref)
