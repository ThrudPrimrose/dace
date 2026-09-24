"""NormalizeMapBody merges some siblings, then refuses a later one and reports no change."""
import dace
from dace.transformation.passes.canonicalize.normalize_map_body import NormalizeMapBody


def copy_body(name: str) -> dace.SDFG:
    sd = dace.SDFG(name)
    sd.add_array('x', [1], dace.float64)
    sd.add_array('y', [1], dace.float64)
    st = sd.add_state()
    tk = st.add_tasklet('t', {'a'}, {'b'}, 'b = a + 1')
    st.add_edge(st.add_read('x'), None, tk, 'a', dace.Memlet('x[0]'))
    st.add_edge(tk, 'b', st.add_write('y'), None, dace.Memlet('y[0]'))
    return sd


sdfg = dace.SDFG('partial_refusal')
for name in 'XYAC':
    sdfg.add_array(name, [8], dace.float64)
st = sdfg.add_state()
me, mx = st.add_map('m', dict(i='0:8'))
bodies = [st.add_nested_sdfg(copy_body(f'n{j}'), {'x'}, {'y'}) for j in range(3)]
# n0: X -> Y, n1: A -> C, n2: A -> X (writes what n0 reads: the reverse direction the pass refuses)
for node, src, dst in zip(bodies, 'XAA', 'YCX'):
    st.add_memlet_path(st.add_read(src), me, node, dst_conn='x', memlet=dace.Memlet(f'{src}[i]'))
    st.add_memlet_path(node, mx, st.add_write(dst), src_conn='y', memlet=dace.Memlet(f'{dst}[i]'))
sdfg.validate()

count = lambda: sum(isinstance(n, dace.nodes.NestedSDFG) for n in st.nodes())
before = sdfg.to_json()
print('nested SDFGs before:', count())
result = NormalizeMapBody().apply_pass(sdfg, {})
print('pass returned:', result)
print('nested SDFGs after:', count())
print('SDFG unchanged:', before == sdfg.to_json())
sdfg.validate()
print('valid')
