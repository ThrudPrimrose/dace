import numpy as np, dace
from drv import *
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import LowerInterstateConditionalAssignmentsToTasklets
N = dace.symbol('N')

@dace.program
def k(a: dace.float64[N], b: dace.float64[N]):
    for j in range(1, N):
        if a[j] > 0.5:
            continue
        b[j] = b[j - 1] + a[j]
s = k.to_sdfg(simplify=True)
print([type(b).__name__ for b in s.all_control_flow_blocks()])
LowerInterstateConditionalAssignmentsToTasklets().apply_pass(s, {})
