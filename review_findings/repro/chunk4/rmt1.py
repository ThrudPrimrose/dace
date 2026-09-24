import dace, numpy as np
from dace.transformation.passes.canonicalize.reverse_map_traversal import ReverseMapTraversal
N = dace.symbol('N')
sdfg = dace.SDFG('rmt1')
sdfg.add_array('a', [N], dace.float64)
sdfg.add_array('b', [N], dace.float64)
st = sdfg.add_state()
# b[N-1-i] = a[N-1-i] + i   (tasklet uses i as a value)
st.add_mapped_tasklet('m', {'i': '0:N'}, {'x': dace.Memlet('a[N-1-i]')}, 'y = x + i',
                      {'y': dace.Memlet('b[N-1-i]')}, external_edges=True)
sdfg.validate()
n = 7
a = np.arange(n, dtype=np.float64); b = np.zeros(n)
ref = np.zeros(n)
for i in range(n): ref[n-1-i] = a[n-1-i] + i
print('applied', ReverseMapTraversal().apply_pass(sdfg, {}))
for node in st.nodes():
    if isinstance(node, dace.nodes.Tasklet): print(node.code.as_string)
    if isinstance(node, dace.nodes.MapEntry): print(node.map.params, node.map.range)
for e in st.edges(): print(e.data)
sdfg(a=a, b=b, N=n)
print(b, ref, np.allclose(b, ref))
