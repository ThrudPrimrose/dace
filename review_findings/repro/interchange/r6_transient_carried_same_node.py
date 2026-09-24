"""MapLoopInterchange misses a body transient carried between loop iterations through a written node.

Loop body (one state):  W: tmp[t % 2] = t + 1 (and tmp[1] = 0 when t == 0)  -> tmp -> R: B[0] += tmp[(t + 1) % 2]
R reads the element W wrote in the previous iteration. `upward_exposed_reads` only asks whether the
read node has an in-edge, not whether the element read was written this iteration, so the carry is missed.
Once the loop is outside the map, every iteration is a fresh call of the nested SDFG with a fresh tmp.
"""
import copy
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import MapLoopInterchange


def build() -> dace.SDFG:
    inner = dace.SDFG('body')
    inner.add_array('B', [1], dace.float64)
    inner.add_array('tmp', [2], dace.float64, transient=True, storage=dace.StorageType.CPU_Heap,
                    lifetime=dace.AllocationLifetime.Scope)
    loop = LoopRegion('tloop', 't < 3', 't', 't = 0', 't = t + 1')
    inner.add_node(loop, is_start_block=True)
    st = loop.add_state('st', is_start_block=True)
    w = st.add_tasklet('W', {}, {'o'}, 'o[t % 2] = t + 1\nif t == 0:\n    o[1] = 0', language=dace.Language.Python)
    tmp = st.add_access('tmp')
    st.add_edge(w, 'o', tmp, None, dace.Memlet('tmp[0:2]', dynamic=True))
    r = st.add_tasklet('R', {'x', 'b'}, {'y'}, 'y = b + x[(t + 1) % 2]')
    st.add_edge(tmp, None, r, 'x', dace.Memlet('tmp[0:2]'))
    st.add_edge(st.add_access('B'), None, r, 'b', dace.Memlet('B[0]'))
    st.add_edge(r, 'y', st.add_access('B'), None, dace.Memlet('B[0]'))

    sdfg = dace.SDFG('transient_carried')
    sdfg.add_array('B', ['N'], dace.float64)
    state = sdfg.add_state('main')
    me, mx = state.add_map('m', {'i': '0:N'})
    ns = state.add_nested_sdfg(inner, {'B'}, {'B'}, {})
    state.add_memlet_path(state.add_access('B'), me, ns, dst_conn='B', memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(ns, mx, state.add_access('B'), src_conn='B', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


sdfg = build()
reference = copy.deepcopy(sdfg)
print('applied:', sdfg.apply_transformations(MapLoopInterchange))
sdfg.validate()
def run(g):
    B = np.zeros(2)
    g(B=B, N=2)
    return B
print('reference:', run(reference))   # 0 + 1 + 2 = 3
print('interchanged:', run(sdfg))
