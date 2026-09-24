"""SinkStateIntoLoop ignores symbols: the sunk state reads 'i', which loop2 redefines per iteration."""
import dace, numpy as np
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.sink_state_into_loop import SinkStateIntoLoop

N = dace.symbol('N')

def build(name):
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_array('C', [1], dace.int64)
    s0 = sdfg.add_state('s0', is_start_block=True)
    l1 = LoopRegion('l1', 'i < 5', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(l1)
    b1 = l1.add_state('b1', is_start_block=True)
    t = b1.add_tasklet('t1', {}, {'o'}, 'o = 1.0')
    b1.add_edge(t, 'o', b1.add_access('A'), None, dace.Memlet('A[i]'))
    st = sdfg.add_state('between')
    t = st.add_tasklet('ri', {}, {'o'}, 'o = i')
    st.add_edge(t, 'o', st.add_access('C'), None, dace.Memlet('C[0]'))
    l2 = LoopRegion('l2', 'i < 5', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(l2)
    b2 = l2.add_state('b2', is_start_block=True)
    t = b2.add_tasklet('t2', {}, {'o'}, 'o = 2.0')
    b2.add_edge(t, 'o', b2.add_access('B'), None, dace.Memlet('B[i]'))
    sdfg.add_edge(s0, l1, dace.InterstateEdge())
    sdfg.add_edge(l1, st, dace.InterstateEdge())
    sdfg.add_edge(st, l2, dace.InterstateEdge())
    sdfg.validate()
    return sdfg

def run(sdfg, n=5):
    A, B, C = np.zeros(n), np.zeros(n), np.zeros(1, np.int64)
    sdfg(A=A, B=B, C=C, N=n)
    return C

sdfg = build('sink1')
print('sunk', SinkStateIntoLoop().apply_pass(sdfg, {}))
print('ref C', run(build('sink1_ref')), 'got C', run(sdfg))
