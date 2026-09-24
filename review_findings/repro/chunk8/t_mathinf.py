import math, numpy as np, dace
from drv import *
N = 16
@dace.program
def f(A: dace.float64[N], B: dace.float64[N]):
    for i in dace.map[0:N]:
        B[i] = min(A[i], math.inf) + math.pow(A[i], 2.0)
s = f.to_sdfg(simplify=True)
for n,_ in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.Tasklet): print(repr(n.code.as_string))
from dace.transformation.passes.vectorization.tasklet_preprocessing_passes import RemoveMathCall
