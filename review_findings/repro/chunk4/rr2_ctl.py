"""RerollUnrolledLoops: overlapping lanes (step < m*g) + zero-trip loop runs extra iterations."""
import dace, numpy as np
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.reroll_unrolled_loops import RerollUnrolledLoops
RerollUnrolledLoops.apply_pass = lambda self, sdfg, _: None

N = dace.symbol('N')

@dace.program
def rr2c(a: dace.float64[N], b: dace.float64[N]):
    for i in range(0, N - 2, 2):
        a[i] = b[i]
        a[i + 1] = b[i + 1]
        a[i + 2] = b[i + 2]

def ref(n):
    a = np.zeros(n); b = np.arange(1, n + 1, dtype=np.float64)
    for i in range(0, n - 2, 2):
        a[i] = b[i]; a[i + 1] = b[i + 1]; a[i + 2] = b[i + 2]
    return a

sdfg = rr2c.to_sdfg(simplify=True)
canonicalize(sdfg)
for n in (1, 2):
    a = np.zeros(n); b = np.arange(1, n + 1, dtype=np.float64)
    sdfg(a=a, b=b, N=n)
    print(n, 'ref', ref(n), 'got', a, 'match', np.allclose(a, ref(n)))
