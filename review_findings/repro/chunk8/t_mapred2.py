import copy, numpy as np, dace
from drv import vec, ntile
import hook, dump
from dace.transformation.passes.vectorization.utils.reductions import recognize_map_reduction
N = 16
@dace.program
def upd(x: dace.float64[N], cols: dace.int32[N], y: dace.float64[N]):
    y[0] = 0.0
    for j in dace.map[0:N]:
        y[j] = y[j] + x[cols[j]]

s = upd.to_sdfg(simplify=True)
dump.dump_nested(s, only_nested=False)
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.MapEntry):
        info = recognize_map_reduction(g, n)
        print('recognize ->', None if info is None else (info.op, info.accumulator, str(info.read_edge.data)))
ref = copy.deepcopy(s); ref.name = 'upd2_ref'
vec(s)
print('Reduce nodes', sum(isinstance(n, dace.libraries.standard.Reduce) for n, _ in s.all_nodes_recursive()), 'tiles', ntile(s))
x = np.arange(1, N + 1, dtype=np.float64); cols = np.arange(N, dtype=np.int32)[::-1].copy()
exp = np.full(N, 100.0); exp[0] = 0; exp += x[cols]
y = np.full(N, 100.0); s(x=x, cols=cols, y=y)
print('vectorized ok', np.allclose(y, exp)); print(y)
