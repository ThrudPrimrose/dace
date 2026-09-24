import dace
from dace.transformation.passes.canonicalize import canonicalize
N = dace.symbol('N'); M = dace.symbol('M')
@dace.program
def minmax(A: dace.float64[N], C: dace.float64[N]):
    for i in dace.map[0:N]:
        C[i] = A[i] + (min(N, M) + max(N, M))
s = minmax.to_sdfg(simplify=True); canonicalize(s, validate=True)
from dace.transformation.passes.split_tasklets import SplitTasklets
SplitTasklets().apply_pass(s, {})
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet): print(repr(n.code.as_string), dict(n.in_connectors))
