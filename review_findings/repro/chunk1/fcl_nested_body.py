"""FuseConsecutiveLoops keys a NestedSDFG only by its type name, so two different bodies compare equal."""
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import FuseConsecutiveLoops


def scalar_kernel(name: str, code: str) -> dace.SDFG:
    inner = dace.SDFG(name)
    inner.add_array('x', (1, ), dace.float64)
    inner.add_array('y', (1, ), dace.float64)
    st = inner.add_state()
    t = st.add_tasklet('t', {'v': None}, {'o': None}, code)
    st.add_edge(st.add_read('x'), None, t, 'v', dace.Memlet('x[0]'))
    st.add_edge(t, 'o', st.add_write('y'), None, dace.Memlet('y[0]'))
    return inner


def loop(sdfg: dace.SDFG, name: str, var: str, begin: str, end: str, body: dace.SDFG) -> LoopRegion:
    region = LoopRegion(name, f'{var} < {end}', var, f'{var} = {begin}', f'{var} = {var} + 1', sdfg=sdfg)
    st = region.add_state(f'{name}_body', is_start_block=True)
    nsdfg = st.add_nested_sdfg(body, {'x': None}, {'y': None})
    st.add_edge(st.add_read('a'), None, nsdfg, 'x', dace.Memlet(f'a[{var}]'))
    st.add_edge(nsdfg, 'y', st.add_write('b'), None, dace.Memlet(f'b[{var}]'))
    return region


sdfg = dace.SDFG('fcl_nested_body')
sdfg.add_symbol('M', dace.int64)
sdfg.add_array('a', (10, ), dace.float64)
sdfg.add_array('b', (10, ), dace.float64)
first = loop(sdfg, 'first', 'i', '0', 'M', scalar_kernel('times_two', 'o = v * 2.0'))
second = loop(sdfg, 'second', 'j', 'M', '10', scalar_kernel('plus_hundred', 'o = v + 100.0'))
sdfg.add_node(first, is_start_block=True)
sdfg.add_node(second)
sdfg.add_edge(first, second, dace.InterstateEdge())
sdfg.validate()

print('pass result:', FuseConsecutiveLoops().apply_pass(sdfg, {}))
print('loops left:', [b.label for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, LoopRegion)])
a = np.arange(10.0)
b = np.zeros(10)
sdfg(a=a, b=b, M=4)
ref = np.concatenate([a[:4] * 2.0, a[4:] + 100.0])
print('reference', ref)
print('got      ', b)
