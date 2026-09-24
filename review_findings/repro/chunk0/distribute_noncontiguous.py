"""DistributeProducerConsumerLoop emits a non-contiguous merge class ahead of the block between
its members, reordering a producer after its consumer.

for i in range(N):
    B0: t[i] = a[i]
    B1: u[i] = b[i] * 2        (forward producer of u, aligned -> may split from B2)
    B2: a[i] = u[i] + 1        (writes a, which B0 reads -> merges with B0)

Groups {B0, B2}, {B1} are sorted by first member, so the pass emits
    for i: { B0; B2 }   for i: { B1 }
and B2 reads u before B1 has written it.
"""
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.distribute_producer_consumer import DistributeProducerConsumerLoop

N = dace.symbol('N')


def stmt(state, inputs, output, code):
    t = state.add_tasklet('t', set(inputs), {'o'}, code)
    for conn, memlet in inputs.items():
        state.add_edge(state.add_read(memlet.split('[')[0]), None, t, conn, dace.Memlet(memlet))
    state.add_edge(t, 'o', state.add_write(output.split('[')[0]), None, dace.Memlet(output))


def build() -> dace.SDFG:
    sdfg = dace.SDFG('distribute_noncontig')
    for name in 'abtu':
        sdfg.add_array(name, [N], dace.float64)
    loop = LoopRegion('L', 'i < N', 'i', 'i = 0', 'i = i + 1')
    sdfg.add_node(loop, is_start_block=True)
    b0 = loop.add_state('B0', is_start_block=True)
    b1 = loop.add_state('B1')
    b2 = loop.add_state('B2')
    loop.add_edge(b0, b1, dace.InterstateEdge())
    loop.add_edge(b1, b2, dace.InterstateEdge())
    stmt(b0, {'x': 'a[i]'}, 't[i]', 'o = x')
    stmt(b1, {'x': 'b[i]'}, 'u[i]', 'o = x * 2.0')
    stmt(b2, {'x': 'u[i]'}, 'a[i]', 'o = x + 1.0')
    return sdfg


def run(sdfg: dace.SDFG):
    n = 5
    rng = np.random.default_rng(0)
    args = {k: rng.random(n) for k in 'ab'}
    args.update(t=np.zeros(n), u=np.zeros(n))
    sdfg(**args, N=n)
    return args


if __name__ == '__main__':
    ref = run(build())
    sdfg = build()
    print('pass returned', DistributeProducerConsumerLoop().apply_pass(sdfg, {}))
    for loop in sdfg.nodes():
        print('  ', loop.label, [b.label for b in loop.nodes()])
    sdfg.validate()
    got = run(sdfg)
    print('a before pass:', np.round(ref['a'], 3))
    print('a after  pass:', np.round(got['a'], 3))
    print('MISMATCH' if not np.allclose(ref['a'], got['a']) else 'ok')
