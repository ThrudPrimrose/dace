import dace
N = dace.symbol('N')

def build(swap=True):
    sdfg = dace.SDFG('swap_alias_rmw' + ('' if swap else '_ref'))
    sdfg.add_array('a', [N, N], dace.float64)
    sdfg.add_array('b', [N, N], dace.float64)
    st = sdfg.add_state('main')
    inner = dace.SDFG('inner')
    inner.add_array('ia', [N, N], dace.float64)
    inner.add_array('ib', [N, N], dace.float64)
    inner.add_symbol('i', dace.int64)
    inner.add_symbol('j', dace.int64)
    s0 = inner.add_state('s0', is_start_block=True)
    # RMW through one access node of the inout connector ``ib`` -> not inlined by normalize_loop_nests.
    t1 = s0.add_tasklet('t1', {'x'}, {'o'}, 'o = x + 1.0')
    t2 = s0.add_tasklet('t2', {'x'}, {'o'}, 'o = x * 2.0')
    rb = s0.add_access('ib')
    s0.add_edge(s0.add_read('ia'), None, t1, 'x', dace.Memlet('ia[i, j]'))
    s0.add_edge(t1, 'o', rb, None, dace.Memlet('ib[i, j]'))
    s0.add_edge(rb, None, t2, 'x', dace.Memlet('ib[i, j]'))
    s0.add_edge(t2, 'o', s0.add_write('ib'), None, dace.Memlet('ib[i, j]'))
    me, mx = st.add_map('m', dict(i='0:N', j='0:N'))
    mapping = {'i': 'j', 'j': 'i', 'N': 'N'} if swap else {'i': 'i', 'j': 'j', 'N': 'N'}
    ns = st.add_nested_sdfg(inner, {'ia', 'ib'}, {'ib'}, symbol_mapping=mapping)
    st.add_memlet_path(st.add_read('a'), me, ns, dst_conn='ia', memlet=dace.Memlet('a[0:N, 0:N]'))
    st.add_memlet_path(st.add_read('b'), me, ns, dst_conn='ib', memlet=dace.Memlet('b[0:N, 0:N]'))
    st.add_memlet_path(ns, mx, st.add_write('b'), src_conn='ib', memlet=dace.Memlet('b[0:N, 0:N]'))
    sdfg.validate()
    return sdfg
