import copy, json
import numpy as np
import dace
from dace.sdfg import nodes, utils as sdutil
from dace.transformation.interstate import SubgraphFission
N = dace.symbol('N')

@dace.program
def body2(a: dace.float64[N, N], b: dace.float64[N, N], c: dace.float64[N, N]):
    for i, j in dace.map[1:N:2, 2:N-1]:
        t = a[i, j] * 2.0
        for k in range(3):
            t = t + a[i, j]
        b[i, j] = t
        c[i, j] = b[i, j] + t

sdfg = body2.to_sdfg(simplify=True)
(state, entry), = [(s, n) for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry)]
(ns,) = {e.dst for e in state.out_edges(entry)}
blocks = list(sdutil.dfs_topological_sort(ns.sdfg))
print([b.label for b in blocks])
ref = copy.deepcopy(sdfg)
for b in blocks:
    g = copy.deepcopy(sdfg); before = json.dumps(g.to_json(), sort_keys=True, default=str)
    try:
        r = g.apply_transformations(SubgraphFission, options={'cut': b.label})
    except Exception as e:
        print(b.label, 'EXC', type(e).__name__, str(e)[:200]); continue
    print(b.label, 'applied', r, 'unchanged' if json.dumps(g.to_json(), sort_keys=True, default=str) == before else 'changed')
    if r:
        g.validate()
        rng = np.random.default_rng(0)
        A = {k: rng.random((7, 7)) for k in 'abc'}; B = copy.deepcopy(A)
        ref(**A, N=7); g(**B, N=7)
        print('  equal:', all(np.allclose(A[k], B[k]) for k in 'bc'), [str(s) for s in g.arrays['t'].shape])
