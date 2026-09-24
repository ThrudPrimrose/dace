"""sift_imperfect_nests moves the pre->inner edge assignment (k = 3*i) inside the inner body,
but the inner loop's own bounds read k, so they now see the previous outer iteration's k."""
import dace, numpy as np
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.sift_statements_into_perfect_nest import sift_imperfect_nests

N = dace.symbol('N')

def build(name):
    sdfg = dace.SDFG(name)
    sdfg.add_array('x', [N], dace.float64)
    sdfg.add_array('y', [3 * N], dace.float64)
    sdfg.add_symbol('k', dace.int64)
    s0 = sdfg.add_state('s0', is_start_block=True)
    outer = LoopRegion('outer', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(outer)
    sdfg.add_edge(s0, outer, dace.InterstateEdge(assignments={'k': '0'}))
    pre = outer.add_state('pre', is_start_block=True)
    t = pre.add_tasklet('p', {}, {'o'}, 'o = i + 1')
    pre.add_edge(t, 'o', pre.add_access('x'), None, dace.Memlet('x[i]'))
    inner = LoopRegion('inner', 'j < k + 3', 'j', 'j = k', 'j = j + 1')
    outer.add_node(inner)
    outer.add_edge(pre, inner, dace.InterstateEdge(assignments={'k': '3 * i'}))
    body = inner.add_state('body', is_start_block=True)
    t2 = body.add_tasklet('b', {'v'}, {'o'}, 'o = v * 10 + j')
    body.add_edge(body.add_access('x'), None, t2, 'v', dace.Memlet('x[i]'))
    body.add_edge(t2, 'o', body.add_access('y'), None, dace.Memlet('y[j]'))
    sdfg.validate()
    return sdfg

def run(sdfg, n=4):
    x = np.zeros(n); y = np.zeros(3 * n)
    sdfg(x=x, y=y, N=n)
    return y

ref = run(build('sift1_ref'))
sdfg = build('sift1')
print('sifted', sift_imperfect_nests(sdfg))
sdfg.validate()
got = run(sdfg)
print('ref y', ref)
print('got y', got)
print('match', np.allclose(ref, got))
