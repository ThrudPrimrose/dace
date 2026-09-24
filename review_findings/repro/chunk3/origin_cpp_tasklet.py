import dace
from dace.transformation.passes.canonicalize.normalize_loop_and_map_origin import NormalizeLoopAndMapOrigin

sdfg = dace.SDFG('cpp_param')
sdfg.add_array('B', [10], dace.int64)
st = sdfg.add_state()
me, mx = st.add_map('m', dict(i='1:10'))
t = st.add_tasklet('t', {}, {'b'}, 'b = i;', language=dace.Language.CPP)
w = st.add_write('B')
st.add_nedge(me, t, dace.Memlet())
st.add_memlet_path(t, mx, w, src_conn='b', memlet=dace.Memlet('B[i]'))
sdfg.validate()
print('pass returned', NormalizeLoopAndMapOrigin().apply_pass(sdfg, {}))
print('map range:', me.map.range, ' write memlet:', st.in_edges(mx)[0].data, ' tasklet:', repr(t.code.as_string))
