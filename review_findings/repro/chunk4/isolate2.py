"""Helper: capture the SDFG right before/after the first firing of a pass class inside canonicalize."""
import copy

def hook(cls):
    captured = {}
    orig = cls.apply_pass

    def patched(self, sdfg, res):
        before = copy.deepcopy(sdfg)
        ret = orig(self, sdfg, res)
        if ret and 'before' not in captured:
            captured['before'] = before
            captured['after'] = copy.deepcopy(sdfg)
            captured['ret'] = ret
        return ret

    cls.apply_pass = patched
    return captured
