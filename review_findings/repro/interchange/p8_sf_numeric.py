import copy, json
import numpy as np
import dace
from dace.sdfg import nodes, utils as sdutil
from dace.transformation.interstate import SubgraphFission
N = dace.symbol('N')

@dace.program
def offs(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[1:N]:
        t = a[i - 1] * 2.0
        u = a[i] + 1.0
        for k in range(3):
            t = t + u
        b[i] = t
        c[i] = b[i] + t * u + c[i]

sdfg = offs.to_sdfg(simplify=True)
(state, entry), = [(s, n) for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry)]
(ns,) = {e.dst for e in state.out_edges(entry)}
blocks = list(sdutil.dfs_topological_sort(ns.sdfg))
print([b.label for b in blocks])
ref = copy.deepcopy(sdfg)
rng = np.random.default_rng(0)
A0 = {k: rng.random(7) for k in 'abc'}
R = copy.deepcopy(A0); ref(**R, N=7)
for b in blocks:
    g = copy.deepcopy(sdfg); before = json.dumps(g.to_json(), sort_keys=True, default=str)
    try:
        r = g.apply_transformations(SubgraphFission, options={'cut': b.label})
    except Exception as e:
        print(b.label, 'EXC', type(e).__name__, str(e)[:200]); continue
    print(b.label, 'applied', r, 'unchanged' if json.dumps(g.to_json(), sort_keys=True, default=str) == before else 'changed')
    if r:
        g.validate()
        B = copy.deepcopy(A0); g(**B, N=7)
        print('  equal:', all(np.allclose(R[k], B[k]) for k in 'bc'), {k: str(v.shape) for k, v in g.arrays.items() if v.transient})
