import dace
from dace.sdfg.state import LoopRegion
from harness import canon_prefix
from dace.transformation.passes.canonicalize import loop_to_einsum as le
N = dace.symbol('N'); M = dace.symbol('M')
@dace.program
def part_t(A: dace.float64[M, M], B: dace.float64[M, M]):
    for i in range(N):
        for j in range(N):
            B[i, j] = A[j, i]
sdfg = part_t.to_sdfg(simplify=True)
canon_prefix(sdfg, 'loop_to_x')
for r in sdfg.all_control_flow_regions():
    print(type(r).__name__, r.label, [ (type(b).__name__, b.label) for b in r.nodes()])
for n, st in sdfg.all_nodes_recursive():
    if isinstance(n, (dace.nodes.Tasklet, dace.nodes.MapEntry)):
        print(n, n.code.as_string if isinstance(n, dace.nodes.Tasklet) else n.map.range)
for r in sdfg.all_control_flow_regions():
    if isinstance(r, LoopRegion):
        nest = le._nest_of(r); print('nest', nest)
        if nest: print(le._body_value(nest, sdfg))
