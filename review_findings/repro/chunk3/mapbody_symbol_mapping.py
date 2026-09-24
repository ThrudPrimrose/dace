"""NormalizeMapBody keeps keep's binding for an inner symbol the dropped sibling binds differently."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody


def body(name: str) -> dace.SDFG:
    sd = dace.SDFG(name)
    sd.add_symbol('k', dace.int64)
    sd.add_array('y', [1], dace.float64)
    st = sd.add_state()
    tk = st.add_tasklet('t', {}, {'o'}, 'o = k')
    st.add_edge(tk, 'o', st.add_write('y'), None, dace.Memlet('y[0]'))
    return sd


def build() -> dace.SDFG:
    sdfg = dace.SDFG('symmap')
    sdfg.add_array('B', [8], dace.float64)
    sdfg.add_array('C', [8], dace.float64)
    st = sdfg.add_state()
    me, mx = st.add_map('m', dict(i='0:8'))
    for arr, binding in (('B', 'i'), ('C', 'i + 1')):
        node = st.add_nested_sdfg(body(f'body_{arr}'), {}, {'y'}, symbol_mapping={'k': binding})
        st.add_nedge(me, node, dace.Memlet())
        st.add_memlet_path(node, mx, st.add_write(arr), src_conn='y', memlet=dace.Memlet(f'{arr}[i]'))
    sdfg.validate()
    return sdfg


for tag, transform in (('before', False), ('after', True)):
    sdfg = build()
    if transform:
        print('pass returned', NormalizeMapBody().apply_pass(sdfg, {}))
        sdfg.validate()
    sdfg.name = f'symmap_{tag}'
    B = np.zeros(8); C = np.zeros(8)
    sdfg(B=B, C=C)
    print(f'{tag}: C = {C.tolist()}  expected {list(np.arange(1, 9, dtype=float))}')
