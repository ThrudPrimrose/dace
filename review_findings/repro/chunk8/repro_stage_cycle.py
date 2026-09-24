"""StageGlobalArrayThroughScalars, top-level (flat map body) path: a read-only subset of a bridge array
that has no outer source node is sourced from the map's own OUTPUT access node, closing a cycle
(outer_drain -> MapEntry -> ... -> MapExit -> outer_drain). The map is not even a tile candidate
(it holds a C++ tasklet), yet VectorizeCPUMultiDim crashes on it."""
import numpy as np
import dace
from drv import vec
from dace.transformation.passes.vectorization.stage_global_array_through_scalars import StageGlobalArrayThroughScalars

N = 8


def build() -> dace.SDFG:
    sdfg = dace.SDFG('flat_bridge_cpp')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N, 2], dace.float64)
    sdfg.add_array('C', [N], dace.float64)
    sdfg.add_array('D', [N], dace.float64)
    st = sdfg.add_state()
    me, mx = st.add_map('m', dict(i=f'0:{N}'))
    b = st.add_access('B')
    t0 = st.add_tasklet('t0', {'__a'}, {'__out'}, '__out = __a + 1.0')
    t1 = st.add_tasklet('t1', {'__b'}, {'__out'}, '__out = __b * 2.0')
    st.add_memlet_path(st.add_read('A'), me, t0, dst_conn='__a', memlet=dace.Memlet('A[i]'))
    st.add_edge(t0, '__out', b, None, dace.Memlet('B[i, 0]'))
    st.add_edge(b, None, t1, '__b', dace.Memlet('B[i, 1]'))
    st.add_memlet_path(b, mx, st.add_write('B'), memlet=dace.Memlet('B[i, 0]'))
    st.add_memlet_path(t1, mx, st.add_write('C'), src_conn='__out', memlet=dace.Memlet('C[i]'))
    t2 = st.add_tasklet('t2', {}, {'o'}, 'o = 3.0;', language=dace.Language.CPP)
    st.add_edge(me, None, t2, None, dace.Memlet())
    st.add_memlet_path(t2, mx, st.add_write('D'), src_conn='o', memlet=dace.Memlet('D[i]'))
    sdfg.validate()
    return sdfg


if __name__ == '__main__':
    s = build()
    A = np.arange(N, dtype=np.float64); B = np.ones((N, 2)); C = np.zeros(N); D = np.zeros(N)
    s(A=A, B=B, C=C, D=D)
    print('unvectorized runs; C ok', np.allclose(C, 2.0), 'B[:,0] ok', np.allclose(B[:, 0], A + 1))
    s = build()
    print('StageGlobalArrayThroughScalars ->', StageGlobalArrayThroughScalars().apply_pass(s, {}))
    try:
        s.validate()
        print('valid')
    except dace.sdfg.InvalidSDFGError as e:
        print('after pass: InvalidSDFGError:', str(e).splitlines()[0])
    s = build()
    try:
        vec(s)
        print('pipeline ok')
    except Exception as e:
        print('VectorizeCPUMultiDim raised', type(e).__name__ + ':', str(e).splitlines()[0])
