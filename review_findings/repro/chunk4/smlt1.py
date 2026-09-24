"""ShrinkMapLocalTransients resizes a transient whose column slice is passed into a NestedSDFG,
leaving the nested SDFG's stride-N view of it stale."""
import dace, numpy as np
from dace.transformation.passes.canonicalize.shrink_map_local_transients import ShrinkMapLocalTransients

N = dace.symbol('N')

def inner(name, scale):
    s = dace.SDFG(name)
    s.add_array('inp', [N], dace.float64, strides=[N])
    s.add_array('out', [N], dace.float64, strides=[N])
    st = s.add_state()
    st.add_mapped_tasklet('k', {'k': '0:N'}, {'x': dace.Memlet('inp[k]')}, f'y = x * {scale}',
                          {'y': dace.Memlet('out[k]')}, external_edges=True)
    return s

def build():
    sdfg = dace.SDFG('smlt1')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N, N], dace.float64)
    sdfg.add_transient('tmp', [N, N], dace.float64)
    st = sdfg.add_state()
    me, mx = st.add_map('col', {'j': '0:N'})
    n1 = st.add_nested_sdfg(inner('n1', 2), {'inp'}, {'out'}, {'N': 'N'})
    n2 = st.add_nested_sdfg(inner('n2', 3), {'inp'}, {'out'}, {'N': 'N'})
    a, b, t = st.add_read('A'), st.add_write('B'), st.add_access('tmp')
    st.add_memlet_path(a, me, n1, dst_conn='inp', memlet=dace.Memlet('A[0:N, j]'))
    st.add_edge(n1, 'out', t, None, dace.Memlet('tmp[0:N, j]'))
    st.add_edge(t, None, n2, 'inp', dace.Memlet('tmp[0:N, j]'))
    st.add_memlet_path(n2, mx, b, src_conn='out', memlet=dace.Memlet('B[0:N, j]'))
    sdfg.validate()
    return sdfg

n = 6
A = np.random.default_rng(0).random((n, n))
ref = A * 6
sdfg = build()
print('desc before:', sdfg.arrays['tmp'].shape, sdfg.arrays['tmp'].strides)
print('pass returned', ShrinkMapLocalTransients().apply_pass(sdfg, {}))
print('desc after:', sdfg.arrays['tmp'].shape, sdfg.arrays['tmp'].strides)
for nd in sdfg.all_nodes_recursive():
    if isinstance(nd[0], dace.nodes.NestedSDFG):
        print(nd[0].label, 'inner strides out/inp:', nd[0].sdfg.arrays['out'].strides, nd[0].sdfg.arrays['inp'].strides)
sdfg.validate()
B = np.zeros((n, n))
sdfg(A=A, B=B, N=n)
print('match', np.allclose(B, ref))
