import numpy as np, dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG
from dace.transformation.passes.vectorization.lower_ite_to_fp_factor import LowerITEToFpFactor
N = dace.symbol('N')

@dace.program
def k(b: dace.int64[N], c: dace.int64[N]):
    for i in dace.map[0:N]:
        if b[i]:
            c[i] = 10
        else:
            c[i] = 20
s = k.to_sdfg(simplify=True)
canonicalize(s, validate=True)
def dump():
    for n, g in s.all_nodes_recursive():
        if isinstance(n, dace.nodes.Tasklet):
            print("  ", n.label, "|", n.code.as_string, {e.dst_conn: str(e.data) for e in g.in_edges(n)})
    for b in s.all_control_flow_blocks(recursive=True):
        if isinstance(b, dace.sdfg.state.ConditionalBlock):
            print("  CB", [(c.as_string if c else None) for c, _ in b.branches])
    for cfg in s.all_control_flow_regions(recursive=True):
        for e in cfg.edges():
            if e.data.assignments: print("  edge", e.data.assignments)
dump()
print("SW", SameWriteSetIfElseToITECFG().apply_pass(s, {}))
dump()
print("FP", LowerITEToFpFactor().apply_pass(s, {}))
dump()
for sd in s.all_sdfgs_recursive():
    for a, d in sd.arrays.items(): print(sd.name, a, d.dtype, d.shape)
