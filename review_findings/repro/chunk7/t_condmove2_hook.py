import copy
import numpy as np, dace
from drv import *
from dace.transformation.passes.vectorization import same_write_set_if_else_to_ite_cfg as sw
N = dace.symbol('N')

@dace.program
def k(a: dace.float64[N], b: dace.float64[N], d: dace.float64[N]):
    for j in dace.map[0:N]:
        d[j] = a[j] * 3.0
    for i in range(1, N):
        c = a[i] > 0.5
        a[i] = b[i - 1]
        if c:
            b[i] = b[i - 1] + 1.0
        else:
            b[i] = b[i - 1] - 1.0

def refk(a, b, d):
    a = a.copy(); b = b.copy(); d = a * 3.0
    for i in range(1, len(a)):
        c = a[i] > 0.5
        a[i] = b[i - 1]
        b[i] = b[i - 1] + 1.0 if c else b[i - 1] - 1.0
    return {"a": a, "b": b, "d": d}

orig = sw.SameWriteSetIfElseToITECFG.apply_pass
def hooked(self, sdfg, r):
    for cfg in sdfg.all_control_flow_regions(recursive=True):
        for e in cfg.edges():
            if e.data.assignments: print("  E", e.src.label, "->", e.dst.label, e.data.assignments)
    for b_ in sdfg.all_control_flow_blocks(recursive=True):
        if isinstance(b_, dace.sdfg.state.ConditionalBlock): print("  CB", [(x.as_string if x else None) for x, _ in b_.branches])
    res = orig(self, sdfg, r); print("SW returned", res); return res
sw.SameWriteSetIfElseToITECFG.apply_pass = hooked
s0 = k.to_sdfg(simplify=True)
v = vectorize(None, "t_condmove2_hook", sdfg=s0)
