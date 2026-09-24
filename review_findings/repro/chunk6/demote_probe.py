import copy, sys
import numpy as np
import dace
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion
from dace.properties import CodeBlock
from dace.transformation.passes.vectorization.demote_data_reading_interstate_symbols import DemoteDataReadingInterstateSymbols

N = 8

def same_edge_chain():
    sdfg = dace.SDFG('demote_same_edge_chain')
    sdfg.add_array('b', (N,), dace.float64)
    sdfg.add_array('out', (N,), dace.float64)
    sdfg.add_symbol('x', dace.float64)
    sdfg.add_symbol('y', dace.float64)
    s0 = sdfg.add_state('s0', is_start_block=True)
    s1 = sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'x': 'b[3]', 'y': 'x + 1'}))
    t = s1.add_tasklet('w', {}, {'_o'}, '_o = y')
    s1.add_edge(t, '_o', s1.add_access('out'), None, dace.Memlet('out[0]'))
    return sdfg

def loop_bound():
    sdfg = dace.SDFG('demote_loop_bound')
    sdfg.add_array('cnt', (1,), dace.int32)
    sdfg.add_array('out', (N,), dace.float64)
    sdfg.add_symbol('n', dace.int32)
    s0 = sdfg.add_state('s0', is_start_block=True)
    loop = LoopRegion('walk', 'i < n', 'i', 'i = 0', 'i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(s0, loop, dace.InterstateEdge(assignments={'n': 'cnt[0]'}))
    body = loop.add_state('body', is_start_block=True)
    t = body.add_tasklet('one', {}, {'_o'}, '_o = 1.0')
    body.add_edge(t, '_o', body.add_access('out'), None, dace.Memlet('out[i]'))
    return sdfg

def cond_block():
    sdfg = dace.SDFG('demote_cond_block')
    sdfg.add_array('b', (N,), dace.float64)
    sdfg.add_array('out', (N,), dace.float64)
    sdfg.add_symbol('x', dace.float64)
    s0 = sdfg.add_state('s0', is_start_block=True)
    cb = ConditionalBlock('cb', sdfg=sdfg)
    sdfg.add_node(cb)
    sdfg.add_edge(s0, cb, dace.InterstateEdge(assignments={'x': 'b[3]'}))
    r = ControlFlowRegion('then', sdfg=sdfg)
    cb.add_branch(CodeBlock('x > 0'), r)
    st = r.add_state('then_s', is_start_block=True)
    t = st.add_tasklet('one', {}, {'_o'}, '_o = 5.0')
    st.add_edge(t, '_o', st.add_access('out'), None, dace.Memlet('out[1]'))
    return sdfg

def nested_mapping():
    sdfg = dace.SDFG('demote_nested_mapping')
    sdfg.add_array('b', (N,), dace.float64)
    sdfg.add_array('out', (N,), dace.float64)
    sdfg.add_symbol('x', dace.float64)
    s0 = sdfg.add_state('s0', is_start_block=True)
    s1 = sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'x': 'b[3]'}))
    inner = dace.SDFG('inner')
    inner.add_array('o', (N,), dace.float64)
    inner.add_symbol('x', dace.float64)
    ist = inner.add_state('is')
    t = ist.add_tasklet('w', {}, {'_o'}, '_o = x * 2')
    ist.add_edge(t, '_o', ist.add_access('o'), None, dace.Memlet('o[0]'))
    ns = s1.add_nested_sdfg(inner, {}, {'o'}, symbol_mapping={'x': 'x'})
    s1.add_edge(ns, 'o', s1.add_access('out'), None, dace.Memlet('out[0:8]'))
    return sdfg

def run(name, build, args):
    ref = build()
    ref.validate()
    res = {k: copy.deepcopy(v) for k, v in args.items()}
    ref.name = name + '_ref'
    ref(**res)
    sd = build()
    sd.name = name + '_dem'
    ret = DemoteDataReadingInterstateSymbols().apply_pass(sd, {})
    print(name, 'demoted:', ret)
    try:
        sd.validate()
    except Exception as e:
        print(name, 'VALIDATION FAILED:', type(e).__name__, str(e)[:300]); return
    got = {k: copy.deepcopy(v) for k, v in args.items()}
    try:
        sd(**got)
    except Exception as e:
        print(name, 'RUN FAILED:', type(e).__name__, str(e)[:400]); return
    for k in args:
        if not np.array_equal(res[k], got[k]):
            print(name, 'MISMATCH on', k, 'ref', res[k], 'got', got[k])
            return
    print(name, 'OK')

which = sys.argv[1:]
b = np.arange(N, dtype=np.float64) + 1
cases = {
  'same_edge_chain': (same_edge_chain, dict(b=b.copy(), out=np.zeros(N))),
  'loop_bound': (loop_bound, dict(cnt=np.array([5], dtype=np.int32), out=np.zeros(N))),
  'cond_block': (cond_block, dict(b=b.copy(), out=np.zeros(N))),
  'nested_mapping': (nested_mapping, dict(b=b.copy(), out=np.zeros(N))),
}
for k in [w for w in which if w in cases] if which else cases:
    run(k, *cases[k])

def nested_map_range():
    sdfg = dace.SDFG('demote_nested_map_range')
    sdfg.add_array('cnt', (1,), dace.int32)
    sdfg.add_array('out', (N,), dace.float64)
    sdfg.add_symbol('n', dace.int32)
    s0 = sdfg.add_state('s0', is_start_block=True)
    s1 = sdfg.add_state('s1')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'n': 'cnt[0]'}))
    inner = dace.SDFG('inner_rng')
    inner.add_array('o', (N,), dace.float64)
    inner.add_symbol('n', dace.int32)
    ist = inner.add_state('is')
    ist.add_mapped_tasklet('w', {'k': '0:n'}, {}, '_o = 1.0', {'_o': dace.Memlet('o[k]')}, external_edges=True)
    ns = s1.add_nested_sdfg(inner, {}, {'o'}, symbol_mapping={'n': 'n'})
    s1.add_edge(ns, 'o', s1.add_access('out'), None, dace.Memlet('out[0:8]'))
    return sdfg

cases2 = {'nested_map_range': (nested_map_range, dict(cnt=np.array([5], dtype=np.int32), out=np.zeros(N)))}
for k in sys.argv[1:]:
    if k in cases2: run(k, *cases2[k])

def self_index():
    sdfg = dace.SDFG('demote_self_index')
    sdfg.add_array('nxt', (N,), dace.int32)
    sdfg.add_array('out', (N,), dace.float64)
    sdfg.add_symbol('x', dace.int32)
    s0 = sdfg.add_state('s0', is_start_block=True)
    s1 = sdfg.add_state('s1')
    s2 = sdfg.add_state('s2')
    sdfg.add_edge(s0, s1, dace.InterstateEdge(assignments={'x': '1'}))
    sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={'x': 'nxt[x]'}))
    t = s2.add_tasklet('w', {}, {'_o'}, '_o = x')
    s2.add_edge(t, '_o', s2.add_access('out'), None, dace.Memlet('out[0]'))
    return sdfg

cases3 = {'self_index': (self_index, dict(nxt=np.array([3, 5, 0, 1, 2, 4, 6, 7], dtype=np.int32), out=np.zeros(N)))}
for k in sys.argv[1:]:
    if k in cases3: run(k, *cases3[k])
