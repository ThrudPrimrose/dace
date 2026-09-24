"""SplitMapForTileRemainder(assume_even=True) emits its extent guard at the SDFG start block, outside the scope that defines the extent symbol (here the outer map param k) -> the generated C++ does not compile."""
import numpy as np, dace
from drv import vec, ntile
N = dace.symbol('N')
@dace.program
def tri(A: dace.float64[N, N], B: dace.float64[N, N]):
    for k in range(1, N):
        for i in dace.map[0:k]:
            B[k, i] = A[k, i] + 1.0
s = tri.to_sdfg(simplify=True)
vec(s, assume_even=True)
print('tiles', ntile(s))
for sd in s.all_sdfgs_recursive():
    for st in sd.states():
        if st.label.startswith('tile_even'):
            print(sd.name, st.label, [n.code.as_string.splitlines()[0] for n in st.nodes()])
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.MapEntry): print(n.map.label, n.map.range)
n = 9
A = np.random.rand(n, n); B = np.zeros((n, n))
s(A=A, B=B, N=n)
print('ran')
