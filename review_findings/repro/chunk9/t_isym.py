import sys; sys.path.insert(0, '.')
from drv import *
import dace
N = dace.symbol('N')

def build():
    sdfg = dace.SDFG('isym_body')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    st = sdfg.add_state('main')
    inner = dace.SDFG('inner')
    inner.add_array('ia', [N], dace.float64)
    inner.add_array('ib', [N], dace.float64)
    inner.add_symbol('i', dace.int64)
    inner.add_symbol('k', dace.int64)
    s0 = inner.add_state('s0', is_start_block=True)
    s1 = inner.add_state('s1')
    inner.add_edge(s0, s1, dace.InterstateEdge(assignments={'k': 'i * 3 + 1'}))
    t = s1.add_tasklet('t', {'x'}, {'o'}, 'o = x + k')
    s1.add_edge(s1.add_read('ia'), None, t, 'x', dace.Memlet('ia[i]'))
    s1.add_edge(t, 'o', s1.add_write('ib'), None, dace.Memlet('ib[i]'))
    me, mx = st.add_map('m', dict(i='0:N'))
    ns = st.add_nested_sdfg(inner, {'ia'}, {'ib'}, symbol_mapping={'i': 'i', 'N': 'N'})
    st.add_memlet_path(st.add_read('a'), me, ns, dst_conn='ia', memlet=dace.Memlet('a[0:N]'))
    st.add_memlet_path(ns, mx, st.add_write('b'), src_conn='ib', memlet=dace.Memlet('b[0:N]'))
    sdfg.validate()
    return sdfg

class P:
    name = 'isym'
    def to_sdfg(self, simplify=True):
        return build()
n = 29
rng = np.random.default_rng(0)
run(P(), dict(a=rng.random(n), b=np.zeros(n), N=n), show=True)
