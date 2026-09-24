import sys, copy
import numpy as np, dace
from drv import *
from dace.transformation.passes.canonicalize import canonicalize
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

n = 19
rng = np.random.default_rng(0)
a = rng.random(n); b = rng.random(n); d = np.zeros(n)
ref = refk(a, b, d)
s0 = k.to_sdfg(simplify=True)
c = copy.deepcopy(s0); c.name = "t_condmove2_canon"; canonicalize(c, validate=True)
print("canon only:", end=" "); compare(ref, run(c, {"a": a, "b": b, "d": d}, N=n))
v = vectorize(None, "t_condmove2_vec", sdfg=copy.deepcopy(s0))
print("full:", end=" "); compare(ref, run(v, {"a": a, "b": b, "d": d}, N=n))
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG
def dump(s):
    for cfg in s.all_control_flow_regions(recursive=True):
        for e in cfg.edges():
            print("  E", cfg.label, e.src.label, "->", e.dst.label, e.data.assignments)
    for b_ in s.all_control_flow_blocks(recursive=True):
        if isinstance(b_, dace.sdfg.state.ConditionalBlock): print("  CB", [(x.as_string if x else None) for x, _ in b_.branches])
        if isinstance(b_, dace.SDFGState): print("  S", b_.label, [f"{getattr(e.src,'data',e.src.label)}->{getattr(e.dst,'data',e.dst.label)} [{e.data}]" for e in b_.edges()])
dump(c)
print("SW:", SameWriteSetIfElseToITECFG().apply_pass(c, {}))
c.validate(); c.name = "t_condmove2_canon_sw"
dump(c)
print("canon+SW:", end=" "); compare(ref, run(c, {"a": a, "b": b, "d": d}, N=n))
