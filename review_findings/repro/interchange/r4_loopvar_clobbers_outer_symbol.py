"""MapLoopInterchange's name-clash check misses an outer symbol defined by an interstate assignment.

Outer:  init --(t = 7)--> main: map i: nsdfg{ for t in range(3): B[i] += 1 } --> after: C[0] = t
`t` is assigned on an outer edge, so it is not in `outer.symbols`; once the loop moves out, its
`t` overwrites the outer `t` that `after` reads.
"""
import copy
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import MapLoopInterchange


def build() -> dace.SDFG:
    inner = dace.SDFG('body')
    inner.add_array('B', [1], dace.float64)
    loop = LoopRegion('tloop', 't < 3', 't', 't = 0', 't = t + 1')
    inner.add_node(loop, is_start_block=True)
    st0 = loop.add_state('st0', is_start_block=True)
    r, w = st0.add_access('B'), st0.add_access('B')
    tl = st0.add_tasklet('acc', {'x'}, {'y'}, 'y = x + 1')
    st0.add_edge(r, None, tl, 'x', dace.Memlet('B[0]'))
    st0.add_edge(tl, 'y', w, None, dace.Memlet('B[0]'))

    sdfg = dace.SDFG('loopvar_clobber')
    sdfg.add_array('B', ['N'], dace.float64)
    sdfg.add_array('C', [1], dace.float64)
    init = sdfg.add_state('init', is_start_block=True)
    state = sdfg.add_state('main')
    after = sdfg.add_state('after')
    sdfg.add_edge(init, state, dace.InterstateEdge(assignments={'t': '7'}))
    sdfg.add_edge(state, after, dace.InterstateEdge())
    me, mx = state.add_map('m', {'i': '0:N'})
    ns = state.add_nested_sdfg(inner, {'B'}, {'B'}, {})
    r, w = state.add_access('B'), state.add_access('B')
    state.add_memlet_path(r, me, ns, dst_conn='B', memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(ns, mx, w, src_conn='B', memlet=dace.Memlet('B[i]'))
    tl = after.add_tasklet('rd', {}, {'y'}, 'y = t')
    after.add_edge(tl, 'y', after.add_access('C'), None, dace.Memlet('C[0]'))
    sdfg.validate()
    return sdfg


sdfg = build()
print('t in outer.symbols:', 't' in sdfg.symbols)
reference = copy.deepcopy(sdfg)
print('applied:', sdfg.apply_transformations(MapLoopInterchange))
sdfg.validate()
def run(g):
    B, C = np.zeros(2), np.zeros(1)
    g(B=B, C=C, N=2)
    return B, C
print('reference:', run(reference))
print('interchanged:', run(sdfg))
