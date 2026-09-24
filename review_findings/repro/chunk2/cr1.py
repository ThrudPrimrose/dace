import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize.loop_to_conditional_reduce import LoopToConditionalReduce
from harness import run_pass_checked, canon_prefix, run_until
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def acc_guard(a: dace.float64[N], b: dace.float64[1]):
    s = 0.0
    for i in range(N):
        if s < 3.0:
            s = s + a[i]
    b[0] = s


@dace.program
def mid_write(a: dace.float64[N], c: dace.float64[N], b: dace.float64[1]):
    s = 0.0
    for i in range(N):
        if a[i] > 0.0:
            c[i] = a[i] * 2.0
            s = s + c[i]
    b[0] = s


def ref_acc_guard(a):
    s = 0.0
    for x in a:
        if s < 3.0:
            s += x
    return s


mode = sys.argv[1]
which = sys.argv[2]
n = 16
a = np.linspace(-1.0, 2.0, n)
prog = {'acc_guard': acc_guard, 'mid_write': mid_write}[which]
sdfg = prog.to_sdfg(simplify=True)
sdfg.name = f'cr1_{which}_{mode}'
if mode == 'direct':
    run_pass_checked(LoopToConditionalReduce(), sdfg)
elif mode == 'prefix':
    run_until(sdfg, LoopToConditionalReduce)
    run_pass_checked(LoopToConditionalReduce(), sdfg)
else:
    canonicalize(sdfg)
sdfg.validate()
for n_, _ in sdfg.all_nodes_recursive():
    if isinstance(n_, dace.nodes.Tasklet) and 'if' in n_.code.as_string:
        print('mask tasklet:', n_.code.as_string.strip(), list(n_.in_connectors))
b = np.zeros(1)
if which == 'acc_guard':
    sdfg(a=a, b=b, N=n)
    print('got', b[0], 'expected', ref_acc_guard(a))
else:
    c = np.zeros(n)
    sdfg(a=a, c=c, b=b, N=n)
    cr = np.where(a > 0, a * 2, 0.0)
    print('b got', b[0], 'expected', cr.sum(), '; c ok:', np.allclose(c, cr), c[:4])
