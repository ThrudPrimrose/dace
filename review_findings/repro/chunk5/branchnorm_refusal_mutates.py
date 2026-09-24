# BranchNormalization._try_rewrite hoists an arm's entry-state symbol binding onto the
# ConditionalBlock's in-edges BEFORE deciding whether the arm can be lowered at all. When the arm
# is then refused (here: its body holds a Map), the pass reports None ("nothing changed") although
# the SDFG was restructured -- the arm's entry state is gone and its binding moved outside.
import json
import dace
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.properties import CodeBlock
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization


def build():
    sdfg = dace.SDFG('bn_refusal_mutates')
    sdfg.add_array('a', [8], dace.float64)
    sdfg.add_array('b', [8], dace.float64)
    sdfg.add_symbol('z1', dace.int64)
    sdfg.add_symbol('c', dace.int64)
    init = sdfg.add_state('init', is_start_block=True)
    cb = ConditionalBlock('cb', sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb)
    arm = ControlFlowRegion('arm', sdfg=sdfg)
    entry = arm.add_state('entry', is_start_block=True)
    body = arm.add_state('body')
    arm.add_edge(entry, body, dace.InterstateEdge(assignments={'__sym_z1': 'z1'}))
    # A Map in the arm: _normalize_single_arm only accepts AccessNode/Tasklet states -> refuses.
    body.add_mapped_tasklet('m', {'j': '0:8'}, {'_in': dace.Memlet('a[j]')},
                            '_out = _in + __sym_z1', {'_out': dace.Memlet('b[j]')},
                            external_edges=True)
    cb.add_branch(CodeBlock('c > 0'), arm)
    sdfg.add_edge(init, cb, dace.InterstateEdge())
    return sdfg


sdfg = build()
sdfg.validate()
before = json.dumps(sdfg.to_json(), sort_keys=True)
arm_states_before = [s.label for s in sdfg.all_states()]
res = BranchNormalization().apply_pass(sdfg, {})
after = json.dumps(sdfg.to_json(), sort_keys=True)
print('BranchNormalization returned', res)
print('states before:', arm_states_before, ' after:', [s.label for s in sdfg.all_states()])
print('in-edge assignments after:', [e.data.assignments for e in sdfg.in_edges(sdfg.node(1))])
print('ConditionalBlock still present:', any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks()))
print('SDFG unchanged:', before == after)
