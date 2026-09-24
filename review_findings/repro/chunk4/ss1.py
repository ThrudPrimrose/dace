"""SplitStatements orders the split by index offset, ignoring the loop variable's (negative) coefficient."""
import dace, numpy as np
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')

@dace.program
def ss1(a: dace.float64[N + 2], b: dace.float64[N + 2], x: dace.float64[N + 2]):
    for i in range(1, N):
        a[N - i] = x[i]
        b[i] = a[N - i + 1]

def ref(a, b, x, n):
    for i in range(1, n):
        a[n - i] = x[i]
        b[i] = a[n - i + 1]

n = 8
rng = np.random.default_rng(0)
a0 = rng.random(n + 2); x = rng.random(n + 2)
ra, rb = a0.copy(), np.zeros(n + 2); ref(ra, rb, x, n)
sdfg = ss1.to_sdfg(simplify=True)
canonicalize(sdfg)
ga, gb = a0.copy(), np.zeros(n + 2)
sdfg(a=ga, b=gb, x=x, N=n)
print('a match', np.allclose(ga, ra), 'b match', np.allclose(gb, rb))
print('ref b', rb)
print('got b', gb)
