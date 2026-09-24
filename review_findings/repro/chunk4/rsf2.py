"""ReorderStateForLoopFusion ignores symbols: 'between' reads symbol k that loop2's body reassigns."""
import dace, numpy as np
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.reorder_state_for_loop_fusion import ReorderStateForLoopFusion
from dace.transformation import pass_pipeline as ppl

N = dace.symbol('N')

def build():
    sdfg = dace.SDFG('rsf2')
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('C', [1], dace.int64)
    sdfg.add_symbol('k', dace.int64)
    s0 = sdfg.add_state('s0', is_start_block=True)
    l1 = LoopRegion('l1', 'i < N', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(l1)
    b1 = l1.add_state('b1', is_start_block=True)
    b1.add_mapped_tasklet('t1', {'z': '0:1'}, {'p': dace.Memlet('A[i-1]')}, 'q = p + 1', {'q': dace.Memlet('A[i]')},
                          external_edges=True)
    st = sdfg.add_state('between')
    t = st.add_tasklet('rk', {}, {'o'}, 'o = k')
    st.add_edge(t, 'o', st.add_access('C'), None, dace.Memlet('C[0]'))
    l2 = LoopRegion('l2', 'i < N', 'i', 'i = 1', 'i = i + 1')
    sdfg.add_node(l2)
    e2 = l2.add_state('e2', is_start_block=True)
    b2 = l2.add_state('b2')
    l2.add_edge(e2, b2, dace.InterstateEdge(assignments={'k': 'i * 10'}))
    b2.add_mapped_tasklet('t2', {'z': '0:1'}, {'p': dace.Memlet('B[i-1]')}, 'q = p + k', {'q': dace.Memlet('B[i]')},
                          external_edges=True)
    s3 = sdfg.add_state('end')
    sdfg.add_edge(s0, l1, dace.InterstateEdge(assignments={'k': '3'}))
    sdfg.add_edge(l1, st, dace.InterstateEdge())
    sdfg.add_edge(st, l2, dace.InterstateEdge())
    sdfg.add_edge(l2, s3, dace.InterstateEdge())
    sdfg.validate()
    return sdfg

def run(sdfg):
    n = 6
    A = np.zeros(n); B = np.zeros(n); C = np.zeros(1, np.int64)
    sdfg(A=A, B=B, C=C, N=n)
    return C

sdfg = build()
ret = ppl.Pipeline([ReorderStateForLoopFusion()]).apply_pass(sdfg, {})
print('pass result', ret['ReorderStateForLoopFusion'])
print('order:', [(e.src.label, e.dst.label) for e in sdfg.edges()])
ref = build(); ref.name = 'rsf2_ref'
print('ref  C', run(ref))
print('pass C', run(sdfg))
