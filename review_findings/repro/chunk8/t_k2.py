import numpy as np, dace, copy, sys
from drv import *
N = dace.symbol('N'); M = dace.symbol('M')
@dace.program
def f(A: dace.float64[N, M], B: dace.float64[N, M]):
    for i, j in dace.map[1:N, 2:M]:
        B[i, j] = A[i, j] * 2.0 + A[i-1, j-2]
strat = sys.argv[1]; widths = tuple(int(x) for x in sys.argv[2].split(','))
kw = {}
if len(sys.argv) > 3: kw['scalar_remainder_emit'] = sys.argv[3]
s = f.to_sdfg(simplify=True)
vec(s, widths=widths, remainder_strategy=strat, **kw)
print("tile ops", ntile(s))
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.MapEntry): print(n.map.label, n.map.range)
ok = True
for (n, m) in [(13, 11), (9, 18), (2, 3), (1, 1), (17, 26)]:
    A = np.random.rand(n, m); B = np.zeros((n, m))
    s(A=A, B=B, N=n, M=m)
    ref = np.zeros((n, m)); ref[1:, 2:] = A[1:, 2:] * 2 + A[:-1, :-2]
    good = np.allclose(B, ref); ok &= good
    print(n, m, "match", good)
