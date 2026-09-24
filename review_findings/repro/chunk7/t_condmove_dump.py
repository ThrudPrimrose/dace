import copy
import numpy as np, dace
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

s0 = k.to_sdfg(simplify=True)
def show(s):
    for sd in s.all_sdfgs_recursive():
        if sd is s: continue
        print("NSDFG", sd.name, {k_: str(v.dtype) + str(v.shape) + ("T" if v.transient else "") for k_, v in sd.arrays.items()})
        for cfg in sd.all_control_flow_regions():
            for e in cfg.edges(): print("  E", e.src.label, "->", e.dst.label, e.data.assignments, e.data.condition.as_string)
        for st in sd.all_states():
            print("  S", st.label, [f"{getattr(e.src,'data',e.src.label)}->{getattr(e.dst,'data',e.dst.label)} [{e.data}]" for e in st.edges()])
show(s0)
from dace.transformation.passes.canonicalize import canonicalize
import sys
if len(sys.argv) > 1:
    canonicalize(s0, validate=True)
else:
    SameWriteSetIfElseToITECFG().apply_pass(s0, {})
print("=== after")
show(s0)
