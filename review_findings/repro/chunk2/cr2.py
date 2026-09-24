import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize.loop_to_conditional_reduce import LoopToConditionalReduce
from harness import run_pass_checked, canon_prefix, run_until
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def live_t(a: dace.float64[N], b: dace.float64[2]):
    s = 0.0
    t = 0.0
    for i in range(N):
        if a[i] > 0.0:
            t = a[i] * 2.0
            s = s + t
    b[0] = s
    b[1] = t


@dace.program
def mid_write_c(a: dace.float64[N], c: dace.float64[1], b: dace.float64[1]):
    s = 0.0
    for i in range(N):
        if a[i] > 0.0:
            c[0] = a[i] * 2.0
            s = s + c[0]
    b[0] = s


def ref(which, a):
    s = 0.0
    t = 0.0
    for x in a:
        if x > 0:
            t = x * 2
            s += t
    return s, t


mode = sys.argv[1]
which = sys.argv[2]
n = 16
a = np.linspace(2.0, -1.0, n)  # last positive element is not the last element
prog = {'live_t': live_t, 'mid_write_c': mid_write_c}[which]
sdfg = prog.to_sdfg(simplify=True)
sdfg.name = f'cr2_{which}_{mode}'
if mode == 'direct':
    run_pass_checked(LoopToConditionalReduce(), sdfg)
elif mode == 'prefix':
    run_until(sdfg, LoopToConditionalReduce)
    sdfg.save(f'cr2_{which}_prefix.sdfg')
    run_pass_checked(LoopToConditionalReduce(), sdfg)
else:
    canonicalize(sdfg)
sdfg.validate()
s_ref, t_ref = ref(which, a)
if which == 'live_t':
    b = np.zeros(2)
    sdfg(a=a, b=b, N=n)
    print('got', b, 'expected', [s_ref, t_ref])
else:
    b = np.zeros(1)
    c = np.zeros(1)
    sdfg(a=a, b=b, c=c, N=n)
    print('got b', b, 'c', c, 'expected', s_ref, t_ref)
