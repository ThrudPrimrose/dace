"""LoopToConditionalReduce strips the interstate assignments on the edge into the guard even when the hoisted
branch body still reads the symbol they define."""
import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize.loop_to_conditional_reduce import LoopToConditionalReduce
from dace.transformation.passes.canonicalize import canonicalize
from harness import run_pass_checked, run_until

N = dace.symbol('N')


@dace.program
def strided_guard(a: dace.float64[2 * N], b: dace.float64[1]):
    s = 0.0
    for i in range(N):
        k = 2 * i + 1
        if a[k] > 0.0:
            s = s + a[k] * 3.0
    b[0] = s


mode = sys.argv[1]
sdfg = strided_guard.to_sdfg(simplify=True)
sdfg.name = f'cr3_{mode}'
if mode == 'direct':
    run_pass_checked(LoopToConditionalReduce(), sdfg)
elif mode == 'prefix':
    run_until(sdfg, LoopToConditionalReduce)
    run_pass_checked(LoopToConditionalReduce(), sdfg)
else:
    canonicalize(sdfg)
for r in sdfg.all_control_flow_regions():
    for e in r.edges():
        if e.data.assignments:
            print('  iedge', r.label, e.data.assignments)
sdfg.validate()
n = 8
a = np.linspace(-1, 1, 2 * n)
b = np.zeros(1)
sdfg(a=a, b=b, N=n)
odd = a[1::2]
print('got', b[0], 'expected', 3 * odd[odd > 0].sum())
