import numpy as np
import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.hoist_iv_updates import HoistInductionVariableUpdates

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def cross(y: dace.float64[1], x: dace.float64[1], c: dace.float64[N]):
    for i in range(N):
        x[0] = y[0] * 0.5
        y[0] = c[i]


def reference(y, x, c):
    y, x = y.copy(), x.copy()
    for i in range(len(c)):
        x[0] = y[0] * 0.5
        y[0] = c[i]
    return y, x


sdfg = cross.to_sdfg(simplify=True)
for loop in sdfg.all_control_flow_regions(recursive=True):
    if isinstance(loop, LoopRegion):
        print(loop.label, [(type(b).__name__, [str(n) for n in b.nodes()]) for b in loop.nodes()])
print('direct pass result:', HoistInductionVariableUpdates().apply_pass(sdfg, {}))
c = np.arange(1.0, 6.0)
y, x = np.array([7.0]), np.zeros(1)
ry, rx = reference(y, x, c)
sdfg(y=y, x=x, c=c, N=5)
print('direct:   reference x', rx, 'got', x)
sdfg = cross.to_sdfg(simplify=True)
canonicalize(sdfg, validate=True)
y, x = np.array([7.0]), np.zeros(1)
sdfg(y=y, x=x, c=c, N=5)
print('pipeline: reference x', rx, 'got', x)
