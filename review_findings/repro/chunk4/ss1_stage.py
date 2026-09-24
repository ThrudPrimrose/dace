import dace, numpy as np
import isolate2
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
cap = isolate2.hook(SplitStatements)
from dace.transformation.passes.canonicalize import canonicalize
N = dace.symbol('N')

@dace.program
def ss1s(a: dace.float64[N + 2], b: dace.float64[N + 2], x: dace.float64[N + 2]):
    for i in range(1, N):
        a[N - i] = x[i]
        b[i] = a[N - i + 1]

sdfg = ss1s.to_sdfg(simplify=True)
canonicalize(sdfg)
print('SplitStatements fired:', cap.get('ret'))
n = 8
rng = np.random.default_rng(0)
a0 = rng.random(n + 2); x = rng.random(n + 2)
for tag in ('before', 'after'):
    s = cap[tag]; s.name = 'ss1s_' + tag
    ga, gb = a0.copy(), np.zeros(n + 2)
    s(a=ga, b=gb, x=x, N=n)
    print(tag, 'b', np.round(gb, 4))
