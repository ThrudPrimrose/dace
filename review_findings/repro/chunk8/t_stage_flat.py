import copy, numpy as np, dace, sys
from drv import vec, ntile
import hook, dump
from dace.transformation.passes.vectorization.stage_global_array_through_scalars import StageGlobalArrayThroughScalars
N = 8

def build(cpp: bool) -> dace.SDFG:
    sdfg = dace.SDFG('flat_bridge' + ('_cpp' if cpp else ''))
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
    if cpp:
        t2 = st.add_tasklet('t2', {}, {'o'}, 'o = 3.0;', language=dace.Language.CPP)
        st.add_edge(me, None, t2, None, dace.Memlet())
        st.add_memlet_path(t2, mx, st.add_write('D'), src_conn='o', memlet=dace.Memlet('D[i]'))
    sdfg.validate()
    return sdfg

s = build(True)
dump.dump_nested(s, only_nested=False)
s2 = copy.deepcopy(s)
print('standalone Stage ->', StageGlobalArrayThroughScalars().apply_pass(s2, {}))
dump.dump_nested(s2, only_nested=False)
try:
    s2.validate(); print('valid')
except Exception as e:
    print('INVALID', type(e).__name__, str(e)[:200])
print('=== pipeline')
s3 = build(True)
try:
    vec(s3)
    print('pipeline returned; tiles', ntile(s3))
    s3.validate(); print('valid after pipeline')
except Exception as e:
    print('pipeline raised', type(e).__name__, str(e)[:300])
