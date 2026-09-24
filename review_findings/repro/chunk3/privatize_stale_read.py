"""PrivatizeReductionAccumulator moves the writeback to a later state, but a same-state reader keeps reading arr."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize.privatize_reduction_accumulator import PrivatizeReductionAccumulator


def build() -> dace.SDFG:
    sdfg = dace.SDFG('priv_stale')
    sdfg.add_array('A', [8], dace.float64)
    sdfg.add_array('s', [1], dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    st = sdfg.add_state()
    me, mx = st.add_map('m', dict(i='0:8'))
    t = st.add_tasklet('acc', {'a'}, {'o'}, 'o = a')
    s_node = st.add_access('s')
    st.add_memlet_path(st.add_read('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[i]'))
    st.add_memlet_path(t, mx, s_node, src_conn='o', memlet=dace.Memlet('s[0]', wcr='lambda x, y: x + y'))
    t2 = st.add_tasklet('use', {'x'}, {'y'}, 'y = 2 * x')
    st.add_edge(s_node, None, t2, 'x', dace.Memlet('s[0]'))
    st.add_edge(t2, 'y', st.add_write('out'), None, dace.Memlet('out[0]'))
    sdfg.validate()
    return sdfg


A = np.arange(8, dtype=np.float64)
for tag, transform in (('before', False), ('after', True)):
    sdfg = build()
    if transform:
        print('pass returned', PrivatizeReductionAccumulator().apply_pass(sdfg, {}))
        sdfg.validate()
    sdfg.name = f'priv_stale_{tag}'
    s = np.ones(1); out = np.zeros(1)
    sdfg(A=A, s=s, out=out)
    print(f'{tag}: s = {s[0]}, out = {out[0]}   expected s = {1 + A.sum()}, out = {2 * (1 + A.sum())}')
