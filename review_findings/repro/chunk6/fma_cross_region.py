"""FuseMultiplyAdd removes a transient that a state inside a LoopRegion still uses."""
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.vectorization.fuse_multiply_add import FuseMultiplyAdd

N = 16


def build() -> dace.SDFG:
    sdfg = dace.SDFG('fma_cross_region')
    for name in ('A', 'B', 'C', 'D'):
        sdfg.add_array(name, (N, ), dace.float64)
    sdfg.add_scalar('t', dace.float64, transient=True)
    st = sdfg.add_state('map_state', is_start_block=True)
    me, mx = st.add_map('m', {'i': f'0:{N}'})
    mul = st.add_tasklet('mul', {'__in1': None, '__in2': None}, {'__out': None}, '__out = (__in1 * __in2)')
    add = st.add_tasklet('add', {'__in1': None, '__in2': None}, {'__out': None}, '__out = (__in1 + __in2)')
    t_an = st.add_access('t')
    st.add_memlet_path(st.add_read('A'), me, mul, dst_conn='__in1', memlet=dace.Memlet('A[i]'))
    st.add_memlet_path(st.add_read('B'), me, mul, dst_conn='__in2', memlet=dace.Memlet('B[i]'))
    st.add_edge(mul, '__out', t_an, None, dace.Memlet('t[0]'))
    st.add_edge(t_an, None, add, '__in1', dace.Memlet('t[0]'))
    st.add_memlet_path(st.add_read('A'), me, add, dst_conn='__in2', memlet=dace.Memlet('A[i]'))
    st.add_memlet_path(add, mx, st.add_write('C'), src_conn='__out', memlet=dace.Memlet('C[i]'))
    # A sequential loop that reuses the scratch scalar ``t``.
    loop = LoopRegion('seq', 'j < 16', 'j', 'j = 1', 'j = j + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(st, loop, dace.InterstateEdge())
    body = loop.add_state('body', is_start_block=True)
    t1 = body.add_tasklet('w', {'_d': None}, {'_o': None}, '_o = _d * 2.0')
    body.add_edge(body.add_read('D'), None, t1, '_d', dace.Memlet('D[j - 1]'))
    tw = body.add_access('t')
    body.add_edge(t1, '_o', tw, None, dace.Memlet('t[0]'))
    t2 = body.add_tasklet('r', {'_y': None}, {'_e': None}, '_e = _y + 1.0')
    body.add_edge(tw, None, t2, '_y', dace.Memlet('t[0]'))
    body.add_edge(t2, '_e', body.add_write('D'), None, dace.Memlet('D[j]'))
    return sdfg


sdfg = build()
sdfg.validate()
print('fused:', FuseMultiplyAdd().apply_pass(sdfg, {}))
print("'t' still declared:", 't' in sdfg.arrays)
try:
    sdfg.validate()
    print('validate: OK')
except Exception as e:
    print('validate FAILED:', type(e).__name__, str(e).splitlines()[0])
