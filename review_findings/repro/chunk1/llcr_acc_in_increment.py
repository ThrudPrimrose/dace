"""LiftLoopCarriedReduction: the increment operand itself reads the accumulator connector."""
import numpy as np
import dace
from dace.sdfg import nodes
from dace.transformation.passes.canonicalize import canonicalize


@dace.program
def acc_in_inc(P: dace.float64[4, 4], X: dace.float64[8, 4, 4]):
    for idx in range(8):
        for i, j in dace.map[0:4, 0:4]:
            with dace.tasklet:
                p << P[i, j]
                x << X[idx, i, j]
                o >> P[i, j]
                o = p + p * x


rng = np.random.default_rng(0)
X = rng.random((8, 4, 4))
P0 = rng.random((4, 4))
rP = P0.copy()
for idx in range(8):
    rP = rP + rP * X[idx]
sdfg = acc_in_inc.to_sdfg(simplify=True)
canonicalize(sdfg, validate=True)
for n, st in sdfg.all_nodes_recursive():
    if isinstance(n, nodes.Tasklet):
        print('tasklet', repr(n.code.as_string), 'in', list(n.in_connectors), [str(e.data) for e in st.out_edges(n)])
P = P0.copy()
sdfg(P=P, X=X)
print('P max err', np.abs(P - rP).max())
