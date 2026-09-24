import sys, copy
import numpy as np, dace
from drv import *
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG
N = dace.symbol('N')

@dace.program
def k(a: dace.float64[N], b: dace.float64[N]):
    for i in dace.map[0:N]:
        c = a[i] > 0.5
        a[i] = 0.0
        if c:
            b[i] = 1.0
        else:
            b[i] = 2.0

n = 19
rng = np.random.default_rng(0)
a = rng.random(n); b = np.zeros(n)
ref = {"a": np.zeros(n), "b": np.where(a > 0.5, 1.0, 2.0)}
s0 = k.to_sdfg(simplify=True)
c = copy.deepcopy(s0); canonicalize(c, validate=True)
def dump(s):
    for cfg in s.all_control_flow_regions(recursive=True):
        for e in cfg.edges():
            if e.data.assignments: print("  edge", e.src.label, "->", e.dst.label, e.data.assignments)
    for b_ in s.all_control_flow_blocks(recursive=True):
        if isinstance(b_, dace.sdfg.state.ConditionalBlock): print("  CB", [(x.as_string if x else None) for x, _ in b_.branches])
dump(c)
c.name = "t_condmove_canon"
print("canon only:", end=" "); compare(ref, run(c, {"a": a, "b": b}, N=n))
print("SW:", SameWriteSetIfElseToITECFG().apply_pass(c, {}))
c.validate()
c.name = "t_condmove_sw"
print("sw:", end=" "); compare(ref, run(c, {"a": a, "b": b}, N=n))
v = vectorize(None, "t_condmove_vec", sdfg=copy.deepcopy(s0))
print("full:", end=" "); compare(ref, run(v, {"a": a, "b": b}, N=n))
p = copy.deepcopy(s0); p.name = "t_condmove_plain"
dump(p)
print("plain:", end=" "); compare(ref, run(p, {"a": a, "b": b}, N=n))
print("SW:", SameWriteSetIfElseToITECFG().apply_pass(p, {}))
p.name = "t_condmove_plain_sw"; p.validate()
dump(p)
print("plain+sw:", end=" "); compare(ref, run(p, {"a": a, "b": b}, N=n))
v = vectorize(None, "t_condmove_vec_nc", sdfg=copy.deepcopy(s0), canon=False)
print("full nocanon:", end=" "); compare(ref, run(v, {"a": a, "b": b}, N=n))
