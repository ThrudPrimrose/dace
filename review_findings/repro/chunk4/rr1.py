"""RerollUnrolledLoops treats lanes whose tasklet reads the loop variable as identical."""
import dace, numpy as np
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')

@dace.program
def rr1(a: dace.float64[N]):
    for i in range(0, N, 2):
        a[i] = i
        a[i + 1] = i

n = 8
ref = np.zeros(n)
for i in range(0, n, 2):
    ref[i] = i; ref[i + 1] = i
sdfg = rr1.to_sdfg(simplify=True)
canonicalize(sdfg)
a = np.zeros(n)
sdfg(a=a, N=n)
print('ref', ref)
print('got', a)
print('match', np.allclose(a, ref))
