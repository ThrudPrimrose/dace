"""CascadeInterstateEdgeAssignmentsUp overwrites a different-valued pre-loop assignment of the
same symbol on the loop's in-edge, and ignores reads of the symbol on interstate edges.

init -[m = 0]-> L(for i in 0..N) { s0 -[j = m + 1]-> s1 (A[i] = j) -[m = K + 1]-> s2 } -> end

Iteration 0 reads j = 0 + 1 = 1; later iterations read j = K + 2.
"""
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.cascade_iedge_assignments_up import CascadeInterstateEdgeAssignmentsUp

N, K = dace.symbol('N'), dace.symbol('K')


def build() -> dace.SDFG:
    sdfg = dace.SDFG('cascade_overwrite')
    sdfg.add_array('A', [N], dace.int64)
    sdfg.add_symbol('K', dace.int64)
    sdfg.add_symbol('m', dace.int64)
    sdfg.add_symbol('j', dace.int64)
    init = sdfg.add_state('init', is_start_block=True)
    loop = LoopRegion('L', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop)
    end = sdfg.add_state('end')
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'m': '0'}))
    sdfg.add_edge(loop, end, dace.InterstateEdge())
    s0 = loop.add_state('s0', is_start_block=True)
    s1 = loop.add_state('s1')
    s2 = loop.add_state('s2')
    loop.add_edge(s0, s1, dace.InterstateEdge(assignments={'j': 'm + 1'}))
    loop.add_edge(s1, s2, dace.InterstateEdge(assignments={'m': 'K + 1'}))
    t = s1.add_tasklet('w', {}, {'o'}, 'o = j')
    s1.add_edge(t, 'o', s1.add_write('A'), None, dace.Memlet('A[i]'))
    return sdfg


def run(sdfg: dace.SDFG) -> np.ndarray:
    A = np.zeros(4, dtype=np.int64)
    sdfg(A=A, N=4, K=10)
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
