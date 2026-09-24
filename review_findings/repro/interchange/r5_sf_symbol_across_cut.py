"""SubgraphFission accepts a body with a symbol assigned before the cut and read after it, then crashes in apply.

Body of `map i`:  s0: b[i] = a[i] * 2  --(k = 3)-->  s1 (cut): t = a[i] + 1  -->  s2: c[i] = t + k
Nesting s0..s1 turns `k` into a symbol->scalar->symbol output `k = __sym_out_k` written by the first nest;
MapFission refuses that assignment (its scalar is tainted by the map-indexed input `a`), which can_be_applied
never asked because it judged the unsplit body.
"""
import copy
import json
import numpy as np
import dace
from dace.sdfg import nodes
from dace.transformation.interstate import SubgraphFission


def build() -> dace.SDFG:
    inner = dace.SDFG('body')
    for name in 'abc':
        inner.add_array(name, [1], dace.float64)
    inner.add_scalar('t', dace.float64, transient=True)
    inner.add_symbol('k', dace.int64)
    s0 = inner.add_state('s0', is_start_block=True)
    s1 = inner.add_state('s1')
    s2 = inner.add_state('s2')
    inner.add_edge(s0, s1, dace.InterstateEdge(assignments={'k': '3'}))
    inner.add_edge(s1, s2, dace.InterstateEdge())
    tl = s0.add_tasklet('dbl', {'x'}, {'y'}, 'y = x * 2')
    s0.add_edge(s0.add_access('a'), None, tl, 'x', dace.Memlet('a[0]'))
    s0.add_edge(tl, 'y', s0.add_access('b'), None, dace.Memlet('b[0]'))
    tl = s1.add_tasklet('inc', {'x'}, {'y'}, 'y = x + 1')
    s1.add_edge(s1.add_access('a'), None, tl, 'x', dace.Memlet('a[0]'))
    s1.add_edge(tl, 'y', s1.add_access('t'), None, dace.Memlet('t[0]'))
    tl = s2.add_tasklet('addk', {'x'}, {'y'}, 'y = x + k')
    s2.add_edge(s2.add_access('t'), None, tl, 'x', dace.Memlet('t[0]'))
    s2.add_edge(tl, 'y', s2.add_access('c'), None, dace.Memlet('c[0]'))

    sdfg = dace.SDFG('sf_symbol')
    for name in 'abc':
        sdfg.add_array(name, ['N'], dace.float64)
    state = sdfg.add_state('main')
    me, mx = state.add_map('m', {'i': '0:N'})
    ns = state.add_nested_sdfg(inner, {'a'}, {'b', 'c'}, {})
    state.add_memlet_path(state.add_access('a'), me, ns, dst_conn='a', memlet=dace.Memlet('a[i]'))
    for name in 'bc':
        state.add_memlet_path(ns, mx, state.add_access(name), src_conn=name, memlet=dace.Memlet(f'{name}[i]'))
    sdfg.validate()
    return sdfg



sdfg = build()
(state, ) = sdfg.states()
before = json.dumps(sdfg.to_json(), sort_keys=True, default=str)
try:
    print('applied:', sdfg.apply_transformations(SubgraphFission, options={'cut': 's1'}))
except Exception as ex:
    print('apply raised:', type(ex).__name__, ex)
print('SDFG changed although nothing applied:', json.dumps(sdfg.to_json(), sort_keys=True, default=str) != before)
