import copy
import numpy as np
import dace
from dace.transformation.passes.vectorization.vectorize_multi_dim import normalize_loop_nests
N = dace.symbol('N')

def build():
    sdfg = dace.SDFG('swap_alias')
    sdfg.add_array('a', [N, N], dace.float64)
    sdfg.add_array('b', [N, N], dace.float64)
    st = sdfg.add_state('main')
    inner = dace.SDFG('inner')
    inner.add_array('ia', [N, N], dace.float64)
    inner.add_array('ib', [N, N], dace.float64)
    inner.add_symbol('i', dace.int64)
    inner.add_symbol('j', dace.int64)
    s0 = inner.add_state('s0', is_start_block=True)
    s1 = inner.add_state('s1')
    inner.add_edge(s0, s1, dace.InterstateEdge())
    t = s1.add_tasklet('t', {'x'}, {'o'}, 'o = x + 1.0')
    s1.add_edge(s1.add_read('ia'), None, t, 'x', dace.Memlet('ia[i, j]'))
    s1.add_edge(t, 'o', s1.add_write('ib'), None, dace.Memlet('ib[i, j]'))
    me, mx = st.add_map('m', dict(i='0:N', j='0:N'))
    ns = st.add_nested_sdfg(inner, {'ia'}, {'ib'}, symbol_mapping={'i': 'j', 'j': 'i', 'N': 'N'})
    st.add_memlet_path(st.add_read('a'), me, ns, dst_conn='ia', memlet=dace.Memlet('a[0:N, 0:N]'))
    st.add_memlet_path(ns, mx, st.add_write('b'), src_conn='ib', memlet=dace.Memlet('b[0:N, 0:N]'))
    sdfg.validate()
    return sdfg

n = 5
a = np.random.rand(n, n)
ref = build(); b_ref = np.zeros((n, n)); ref(a=a, b=b_ref, N=n)
s = build(); s.name = 'swap_alias_norm'
normalize_loop_nests(s)
ns = [x for x, _ in s.all_nodes_recursive() if isinstance(x, dace.nodes.NestedSDFG)]
for x in ns:
    print("mapping:", dict(x.symbol_mapping), "inner symbols:", sorted(x.sdfg.symbols), "inner memlets:",
          [str(e.data) for st in x.sdfg.states() for e in st.edges()])
try:
    s.validate(); print("valid")
except Exception as ex:
    print("INVALID:", type(ex).__name__, str(ex)[:200])
b = np.zeros((n, n)); s(a=a, b=b, N=n)
print("match:", np.allclose(b, b_ref))
