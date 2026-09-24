"""MapLoopInterchange misses a symbol the loop body assigns in one iteration and reads in the next.

Body of `for t in range(T)`:  st0: B[i] += s  --(s = t)-->  st1 (empty).
Iteration t reads the s that iteration t-1 assigned (outer s at t=0).
"""
import copy
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import MapLoopInterchange


def build() -> dace.SDFG:
    inner = dace.SDFG('body')
    inner.add_array('B', [1], dace.float64)
    inner.add_symbol('s', dace.int64)
    loop = LoopRegion('tloop', 't < T', 't', 't = 0', 't = t + 1')
    inner.add_node(loop, is_start_block=True)
    st0 = loop.add_state('st0', is_start_block=True)
    st1 = loop.add_state('st1')
    loop.add_edge(st0, st1, dace.InterstateEdge(assignments={'s': 't'}))
    r, w = st0.add_access('B'), st0.add_access('B')
    tl = st0.add_tasklet('acc', {'x'}, {'y'}, 'y = x + s')
    st0.add_edge(r, None, tl, 'x', dace.Memlet('B[0]'))
    st0.add_edge(tl, 'y', w, None, dace.Memlet('B[0]'))

    sdfg = dace.SDFG('carried_symbol')
    sdfg.add_array('B', ['N'], dace.float64)
    sdfg.add_symbol('s', dace.int64)
    sdfg.add_symbol('T', dace.int64)
    state = sdfg.add_state('main')
    me, mx = state.add_map('m', {'i': '0:N'})
    ns = state.add_nested_sdfg(inner, {'B'}, {'B'}, {'s': 's', 'T': 'T'})
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
    g(B=B, N=3, T=4, s=100)
    return B
print('reference:', run(reference))    # 100 + 0 + 1 + 2 = 103
print('interchanged:', run(sdfg))
