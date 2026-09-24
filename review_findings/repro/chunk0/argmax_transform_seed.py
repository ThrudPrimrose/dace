"""ArgMaxLift transform-only path (``x = abs(a[i])``): same unchecked seed drop."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def prog(a: dace.float64[N], out: dace.float64[1]):
    x = 2.0
    for i in range(N):
        if abs(a[i]) > x:
            x = abs(a[i])
    out[0] = x


if __name__ == '__main__':
    a = np.random.default_rng(0).random(8) - 0.5
    sdfg = canonicalize(prog.to_sdfg())
    print('library nodes:', sorted({type(n).__name__ for n, _ in sdfg.all_nodes_recursive()
                                    if isinstance(n, dace.nodes.LibraryNode)}))
    out = np.zeros(1)
    sdfg(a=a, out=out, N=8)
    print('expected [2.0], got', out.tolist())
    print('MISMATCH' if not np.allclose(out, [2.0]) else 'ok')
