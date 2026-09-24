import dace, numpy as np
from dace.transformation.passes.canonicalize.symbol_dedup import SymbolDedup

def build():
    sdfg = dace.SDFG('sd1')
    sdfg.add_symbol('t', dace.int64)
    sdfg.add_symbol('u', dace.int64)
    sdfg.add_array('out', [2], dace.int64)
    s0 = sdfg.add_state('s0', is_start_block=True)
    s1 = sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'t': 't + 1', 'u': 't + 1'}))
    tk = s1.add_tasklet('w', {}, {'o0', 'o1'}, 'o0 = t\no1 = u')
    an = s1.add_access('out')
    s1.add_edge(tk, 'o0', an, None, dace.Memlet('out[0]'))
    s1.add_edge(tk, 'o1', an, None, dace.Memlet('out[1]'))
    return sdfg

a = build(); o = np.zeros(2, np.int64); a(out=o, t=10); print('before', o)
b = build(); b.name = 'sd1b'; print('dedup returned', SymbolDedup().apply_pass(b, {}))
print([e.data.assignments for e in b.all_interstate_edges()])
o = np.zeros(2, np.int64); b(out=o, t=10); print('after', o)
