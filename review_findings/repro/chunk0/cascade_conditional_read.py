"""CascadeInterstateEdgeAssignmentsUp does not see a read of the hoisted symbol inside a
ConditionalBlock predecessor (ConditionalBlock is not a ControlFlowRegion, so
_block_reads_symbols/_block_writes return nothing for it).

m is an SDFG argument symbol. Loop body:
  cb: if m > 5: A[i] = 1 else: A[i] = 2
  cb -> s1 -[m = K + 1]-> s2
Iteration 0 branches on the argument m, later iterations on K + 1.
"""
import numpy as np
import dace
from dace.properties import CodeBlock
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.canonicalize.cascade_iedge_assignments_up import CascadeInterstateEdgeAssignmentsUp

N = dace.symbol('N')


def branch(name: str, value: int) -> ControlFlowRegion:
    region = ControlFlowRegion(name)
    st = region.add_state(name + '_s', is_start_block=True)
    t = st.add_tasklet('w', {}, {'o'}, f'o = {value}')
    st.add_edge(t, 'o', st.add_write('A'), None, dace.Memlet('A[i]'))
    return region


def build() -> dace.SDFG:
    sdfg = dace.SDFG('cascade_cond_read')
    sdfg.add_array('A', [N], dace.int64)
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('m', dace.int64)
    loop = LoopRegion('L', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    cb = ConditionalBlock('cb')
    loop.add_node(cb, is_start_block=True)
    cb.add_branch(CodeBlock('m > 5'), branch('then', 1))
    cb.add_branch(None, branch('else', 2))
    s1 = loop.add_state('s1')
    s2 = loop.add_state('s2')
    loop.add_edge(cb, s1, dace.InterstateEdge())
    loop.add_edge(s1, s2, dace.InterstateEdge(assignments={'m': 'K + 1'}))
    return sdfg


def run(sdfg: dace.SDFG) -> np.ndarray:
    A = np.zeros(4, dtype=np.int64)
    sdfg(A=A, N=4, K=10, m=0)
    return A


if __name__ == '__main__':
    ref = build()
    ref.validate()
    before = run(ref)
    sdfg = build()
    ret = CascadeInterstateEdgeAssignmentsUp().apply_pass(sdfg, {})
    print('pass returned', ret)
    for e in sdfg.all_interstate_edges():
        if e.data.assignments:
            print('  edge', e.src.label, '->', e.dst.label, e.data.assignments)
    sdfg.validate()
    after = run(sdfg)
    print('before pass:', before)
    print('after  pass:', after)
    print('MISMATCH' if not np.array_equal(before, after) else 'ok')
