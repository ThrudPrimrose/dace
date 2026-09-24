"""MapLoopInterchange keeps a loop whose bound symbol the loop body reassigns.

Body of `for (t = 0; t < M; t++)`:  st0: B[i] += 1  --(M = 2)-->  st1 (empty).
Inside the map the loop stops after 2 iterations; outside it the outer M (5) is never reassigned.
"""
import copy
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import MapLoopInterchange


def build() -> dace.SDFG:
    inner = dace.SDFG('body')
    inner.add_array('B', [1], dace.float64)
    inner.add_symbol('M', dace.int64)
    loop = LoopRegion('tloop', 't < M', 't', 't = 0', 't = t + 1')
    inner.add_node(loop, is_start_block=True)
    st0 = loop.add_state('st0', is_start_block=True)
    st1 = loop.add_state('st1')
    loop.add_edge(st0, st1, dace.InterstateEdge(assignments={'M': '2'}))
    r, w = st0.add_access('B'), st0.add_access('B')
    tl = st0.add_tasklet('acc', {'x'}, {'y'}, 'y = x + 1')
    st0.add_edge(r, None, tl, 'x', dace.Memlet('B[0]'))
    st0.add_edge(tl, 'y', w, None, dace.Memlet('B[0]'))

    sdfg = dace.SDFG('bound_reassigned')
    sdfg.add_array('B', ['N'], dace.float64)
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_array('C', ['M'], dace.float64)  # makes M an argument of the outer SDFG
    state = sdfg.add_state('main')
    me, mx = state.add_map('m', {'i': '0:N'})
    ns = state.add_nested_sdfg(inner, {'B'}, {'B'}, {'M': 'M'})
    r, w = state.add_access('B'), state.add_access('B')
    state.add_memlet_path(r, me, ns, dst_conn='B', memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(ns, mx, w, src_conn='B', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


sdfg = build()
reference = copy.deepcopy(sdfg)
print('applied:', sdfg.apply_transformations(MapLoopInterchange))
sdfg.validate()
def run(g):
    B = np.zeros(3)
    g(B=B, C=np.zeros(5), N=3, M=5)
    return B
print('reference:', run(reference))
print('interchanged:', run(sdfg))
