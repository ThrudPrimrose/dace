"""Frontend + full canonicalize for the DistributeProducerConsumerLoop reordering bug."""
import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def prog(a: dace.float64[N], b: dace.float64[N], t: dace.float64[N], u: dace.float64[N]):
    for i in range(N):
        t[i] = a[i]
        if i % 2 == 0:
            u[i] = b[i] * 2.0
        else:
            u[i] = b[i] * 3.0
        a[i] = u[i] + 1.0


def reference(a, b, t, u):
    for i in range(a.shape[0]):
        t[i] = a[i]
        u[i] = b[i] * 2.0 if i % 2 == 0 else b[i] * 3.0
        a[i] = u[i] + 1.0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        from dace.transformation.passes.canonicalize import distribute_producer_consumer as dpc
        dpc.DistributeProducerConsumerLoop.apply_pass = lambda self, sdfg, res: None
    n = 6
    rng = np.random.default_rng(0)
    a0, b = rng.random(n), rng.random(n)
    ra, rt, ru = a0.copy(), np.zeros(n), np.zeros(n)
    reference(ra, b, rt, ru)
    sdfg = canonicalize(prog.to_sdfg())
    a, t, u = a0.copy(), np.zeros(n), np.zeros(n)
    sdfg(a=a, b=b, t=t, u=u, N=n)
    print('numpy a:', np.round(ra, 3))
    print('dace  a:', np.round(a, 3))
    print('MISMATCH' if not (np.allclose(a, ra) and np.allclose(t, rt) and np.allclose(u, ru)) else 'ok')
