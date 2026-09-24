"""SplitStatements: iteration_distinct accepts a non-injective store index (s[i // 2])."""
import dace, numpy as np
import isolate2
from dace.transformation.passes.canonicalize.split_statements import SplitStatements
cap = isolate2.hook(SplitStatements)
from dace.transformation.passes.canonicalize import canonicalize
N = dace.symbol('N')

@dace.program
def ss2(s: dace.float64[N], b: dace.float64[N], x: dace.float64[N]):
    for i in range(N):
        s[i // 2] = s[i // 2] + x[i]
        b[i] = s[i // 2]

def ref(s, b, x, n):
    for i in range(n):
        s[i // 2] = s[i // 2] + x[i]
        b[i] = s[i // 2]

n = 8
x = np.arange(1, n + 1, dtype=np.float64)
rs, rb = np.zeros(n), np.zeros(n); ref(rs, rb, x, n)
sdfg = ss2.to_sdfg(simplify=True)
canonicalize(sdfg)
print('SplitStatements fired:', cap.get('ret'))
gs, gb = np.zeros(n), np.zeros(n)
sdfg(s=gs, b=gb, x=x, N=n)
print('ref b', rb); print('got b', gb); print('s match', np.allclose(gs, rs), 'b match', np.allclose(gb, rb))
if 'before' in cap:
    for tag in ('before', 'after'):
        t = cap[tag]; t.name = 'ss2_' + tag
        gs, gb = np.zeros(n), np.zeros(n)
        t(s=gs, b=gb, x=x, N=n)
        print(tag, 'b', gb)
