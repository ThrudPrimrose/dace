"""SameWriteSetIfElseToITECFG re-evaluates a lifted guard at the merge state, after an intervening write."""
import copy
import numpy as np, dace
from drv import compare, run
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG

N = dace.symbol('N')


@dace.program
def k(a: dace.float64[N], b: dace.float64[N]):
    for i in range(1, N):
        c = a[i] > 0.5
        a[i] = b[i - 1]
        if c:
            b[i] = b[i - 1] + 1.0
        else:
            b[i] = b[i - 1] - 1.0


def reference(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    a, b = a.copy(), b.copy()
    for i in range(1, len(a)):
        c = a[i] > 0.5
        a[i] = b[i - 1]
        b[i] = b[i - 1] + 1.0 if c else b[i - 1] - 1.0
    return {"a": a, "b": b}


n = 19
rng = np.random.default_rng(0)
a, b = rng.random(n), rng.random(n)
ref = reference(a, b)
sdfg = k.to_sdfg(simplify=True)
canonicalize(sdfg, validate=True)
sdfg.name = "rp_sw_canon"
print("canonicalized:", end=" ")
compare(ref, run(sdfg, {"a": a, "b": b}, N=n))
for e in (e for cfg in sdfg.all_control_flow_regions(recursive=True) for e in cfg.edges()):
    if e.data.assignments:
        print("guard symbol edge:", e.src.label, "->", e.dst.label, e.data.assignments)
after = copy.deepcopy(sdfg)
after.name = "rp_sw_after"
print("SameWriteSetIfElseToITECFG returned", SameWriteSetIfElseToITECFG().apply_pass(after, {}))
after.validate()
for st in after.all_states():
    for t in st.nodes():
        if isinstance(t, dace.nodes.Tasklet) and t.label.startswith("lift_cond"):
            print("lifted guard in state", st.label, ":", t.code.as_string, [str(e.data) for e in st.in_edges(t)])
print("after pass:", end=" ")
compare(ref, run(after, {"a": a, "b": b}, N=n))
