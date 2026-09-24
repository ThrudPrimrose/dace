"""Control: rr1 with RerollUnrolledLoops disabled gives the right answer."""
import dace, numpy as np
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.reroll_unrolled_loops import RerollUnrolledLoops
RerollUnrolledLoops.apply_pass = lambda self, sdfg, _: None
N = dace.symbol('N')

@dace.program
def rr1c(a: dace.float64[N]):
    for i in range(0, N, 2):
        a[i] = i
        a[i + 1] = i

n = 8
sdfg = rr1c.to_sdfg(simplify=True)
canonicalize(sdfg)
a = np.zeros(n)
sdfg(a=a, N=n)
print('got (reroll disabled)', a)
