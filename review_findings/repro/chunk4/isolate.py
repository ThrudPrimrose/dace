"""Helper: capture the SDFG right before and right after RerollUnrolledLoops inside canonicalize."""
import copy
from dace.transformation.passes.canonicalize.reroll_unrolled_loops import RerollUnrolledLoops

captured = {}
orig = RerollUnrolledLoops.apply_pass

def patched(self, sdfg, res):
    before = copy.deepcopy(sdfg)
    ret = orig(self, sdfg, res)
    if ret and 'before' not in captured:
        captured['before'] = before
        captured['after'] = copy.deepcopy(sdfg)
        captured['ret'] = ret
    return ret

RerollUnrolledLoops.apply_pass = patched
