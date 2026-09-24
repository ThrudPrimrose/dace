"""NormalizeMapBody binds the consumer's carrier array to the producer's without comparing subsets."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody


def producer() -> dace.SDFG:
    sd = dace.SDFG('producer')
    sd.add_array('a', [2], dace.float64)
    sd.add_array('o', [2], dace.float64)
    st = sd.add_state()
    st.add_mapped_tasklet('p', dict(k='0:2'), dict(x=dace.Memlet('a[k]')), 'y = x * 10', dict(y=dace.Memlet('o[k]')),
                          external_edges=True)
    return sd


def consumer() -> dace.SDFG:
    sd = dace.SDFG('consumer')
    sd.add_array('t', [1], dace.float64)
    sd.add_array('b', [1], dace.float64)
    st = sd.add_state()
    tk = st.add_tasklet('c', {'x'}, {'y'}, 'y = x')
    st.add_edge(st.add_read('t'), None, tk, 'x', dace.Memlet('t[0]'))
    st.add_edge(tk, 'y', st.add_write('b'), None, dace.Memlet('b[0]'))
    return sd


def build() -> dace.SDFG:
    sdfg = dace.SDFG('carrier_subset')
    sdfg.add_array('A', [4, 2], dace.float64)
    sdfg.add_array('B', [4], dace.float64)
    sdfg.add_array('tmp', [2], dace.float64, transient=True)
    st = sdfg.add_state()
    me, mx = st.add_map('m', dict(i='0:4'))
    k = st.add_nested_sdfg(producer(), {'a'}, {'o'})
    d = st.add_nested_sdfg(consumer(), {'t'}, {'b'})
    tmp = st.add_access('tmp')
    st.add_memlet_path(st.add_read('A'), me, k, dst_conn='a', memlet=dace.Memlet('A[i, 0:2]'))
    st.add_edge(k, 'o', tmp, None, dace.Memlet('tmp[0:2]'))
    st.add_edge(tmp, None, d, 't', dace.Memlet('tmp[1]'))
    st.add_memlet_path(d, mx, st.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


A = np.arange(8, dtype=np.float64).reshape(4, 2)
expected = A[:, 1] * 10

sdfg = build()
B = np.zeros(4)
sdfg(A=A, B=B)
print('before pass:', B.tolist(), 'expected', expected.tolist())

sdfg = build()
print('pass returned', NormalizeMapBody().apply_pass(sdfg, {}))
sdfg.validate()
sdfg.name = 'carrier_subset_after'
B = np.zeros(4)
sdfg(A=A, B=B)
print('after pass :', B.tolist(), 'expected', expected.tolist())
