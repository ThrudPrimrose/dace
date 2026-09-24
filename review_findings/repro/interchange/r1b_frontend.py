"""Frontend version: a copy the simplifier fuses into the map's state is repeated per loop iteration."""
import copy
import numpy as np
import dace
from dace.sdfg import nodes
from dace.transformation.interstate import MapLoopInterchange

N, T = dace.symbol('N'), dace.symbol('T')


@dace.program
def prog(B: dace.float64[N], tmp: dace.float64[N]):
    tmp[:] = B
    for i in dace.map[0:N]:
        for t in range(T):
            B[i] = B[i] + tmp[i]


sdfg = prog.to_sdfg(simplify=True)
print([ (s.label, [type(n).__name__ for n in s.nodes()]) for s in sdfg.states()])
reference = copy.deepcopy(sdfg)
print('applied:', sdfg.apply_transformations(MapLoopInterchange))
def run(g):
    B = np.ones(4)
    g(B=B, tmp=np.zeros(4), N=4, T=3)
    return B
print('reference:', run(reference))
print('interchanged:', run(sdfg))
