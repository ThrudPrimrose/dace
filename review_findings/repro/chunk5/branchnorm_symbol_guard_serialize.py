# BranchNormalization.serialize_two_arm turns `if k > 0: A else: B` into `if k > 0: A` followed by
# `if not (k > 0): B`. guard_snapshot_verdict only asks whether an arm writes an ARRAY the guard
# reads; an arm that reassigns the guard's SYMBOL on an interstate edge (k = k - 1) is not seen,
# so the re-test after A flips and B runs as well.
import numpy as np
import dace
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.properties import CodeBlock
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization


def write_one(state, name):
    t = state.add_tasklet(f'set_{name}', {}, {'o'}, 'o = 1.0')
    state.add_edge(t, 'o', state.add_write(name), None, dace.Memlet(f'{name}[0]'))


def build():
    sdfg = dace.SDFG('bn_symbol_guard')
    for name in ('x', 'y', 'z'):
        sdfg.add_array(name, [1], dace.float64)
    sdfg.add_symbol('k', dace.int64)
    init = sdfg.add_state('init', is_start_block=True)
    cb = ConditionalBlock('cb', sdfg=sdfg, parent=sdfg)
    sdfg.add_node(cb)
    arm0 = ControlFlowRegion('arm0', sdfg=sdfg)
    a1 = arm0.add_state('a1', is_start_block=True)
    write_one(a1, 'x')
    a2 = arm0.add_state('a2')
    write_one(a2, 'y')
    arm0.add_edge(a1, a2, dace.InterstateEdge(assignments={'k': 'k - 1'}))
    arm1 = ControlFlowRegion('arm1', sdfg=sdfg)
    b1 = arm1.add_state('b1', is_start_block=True)
    write_one(b1, 'z')
    cb.add_branch(CodeBlock('k > 0'), arm0)
    cb.add_branch(None, arm1)
    # k is a program argument (k=1 at the call): nothing upstream binds it to a constant.
    sdfg.add_edge(init, cb, dace.InterstateEdge())
    return sdfg


def run(sdfg):
    out = []
    for k in (0, 1, 2, 3):
        x, y, z = np.zeros(1), np.zeros(1), np.zeros(1)
        sdfg(x=x, y=y, z=z, k=k)
        out.append((k, x[0], y[0], z[0]))
    return out


ref = build()
ref.validate()
print('original      [(k, x, y, z)] =', run(ref))
sdfg = build()
sdfg.name = 'bn_symbol_guard_lowered'
print('BranchNormalization returned', BranchNormalization().apply_pass(sdfg, {}))
sdfg.validate()
print('after pass    [(k, x, y, z)] =', run(sdfg))
