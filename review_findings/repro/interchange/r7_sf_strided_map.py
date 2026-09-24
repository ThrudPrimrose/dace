"""SubgraphFission accepts a strided map (can_be_applied True) and then crashes in apply, leaving the SDFG half-rewritten.

Same body as the shipped test's `long_body`, only the map range is 1:N:2 instead of 0:N.
"""
import json
import traceback
import dace
from dace.sdfg import nodes, utils as sdutil
from dace.transformation.interstate import SubgraphFission
from dace.transformation.dataflow import map_fission

N = dace.symbol('N')


@dace.program
def strided_body(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[1:N:2]:
        t = a[i] * 2.0
        for k in range(3):
            t = t + a[i]
        b[i] = t
        c[i] = b[i] + t


sdfg = strided_body.to_sdfg(simplify=True)
(state, entry), = [(s, n) for s in sdfg.states() for n in s.nodes() if isinstance(n, nodes.MapEntry)]
(nsdfg, ) = {e.dst for e in state.out_edges(entry)}
print('map range:', entry.map.range)
cut = next(iter(nsdfg.sdfg.start_block.label for _ in [0]))
print('cut after the first body block:', cut)
before = json.dumps(sdfg.to_json(), sort_keys=True, default=str)
orig = map_fission.MapFission.can_be_applied
def spy(self, graph, expr_index, sdfg, permissive=False):
    r = orig(self, graph, expr_index, sdfg, permissive)
    print('  MapFission.can_be_applied ->', r, 'on body blocks', [b.label for b in self.nested_sdfg.sdfg.nodes()])
    return r
map_fission.MapFission.can_be_applied = spy
try:
    print('applied:', sdfg.apply_transformations(SubgraphFission, options={'cut': cut}))
except Exception as ex:
    print('apply raised:', type(ex).__name__, ex)
print('SDFG changed although nothing applied:', json.dumps(sdfg.to_json(), sort_keys=True, default=str) != before)
