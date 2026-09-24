"""FuseConditions: two guards reading the same array element, the first body writes it."""
import numpy as np
import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.canonicalize.fuse_conditions import FuseConditions


@dace.program
def two_guards(a: dace.float64[4], b: dace.float64[4]):
    if a[0] > 0:
        a[0] = -1.0
    if a[0] > 0:
        b[0] = 5.0


for mo in (False, True):
    sdfg = two_guards.to_sdfg(simplify=True)
    before = sum(isinstance(n, ConditionalBlock) for n in sdfg.all_control_flow_blocks(recursive=True))
    res = FuseConditions(matcher_order=mo).apply_pass(sdfg, {})
    after = sum(isinstance(n, ConditionalBlock) for n in sdfg.all_control_flow_blocks(recursive=True))
    print('matcher_order', mo, 'result', res, 'cond blocks', before, '->', after)
    for n in sdfg.all_control_flow_blocks(recursive=True):
        if isinstance(n, ConditionalBlock):
            print('  ', [(c.as_string if c is not None else None) for c, _ in n.branches])
