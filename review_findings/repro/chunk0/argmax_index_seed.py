"""ArgMaxLift (index-carrier path) drops the pre-loop seeds of the value and index carriers
without checking that the seed is ``a[start - 1]``.  A seed above every element must survive
(x = 2.0, idx = -1); the lifted ArgReduce returns max(a) and its position instead.
Pass 'disable' to run the same pipeline with ArgMaxLift turned off."""
import sys
import numpy as np
import dace

if len(sys.argv) > 1:
    from dace.transformation.passes.canonicalize import arg_max_lift
    arg_max_lift.ArgMaxLift.apply_pass = lambda self, sdfg, res: None
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def prog(a: dace.float64[N], out: dace.float64[2]):
    x = 2.0
    idx = -1
    for i in range(N):
        if a[i] > x:
            x = a[i]
            idx = i
    out[0] = x
    out[1] = idx


if __name__ == '__main__':
    a = np.random.default_rng(0).random(8)  # all in [0, 1) < 2.0
    sdfg = canonicalize(prog.to_sdfg())
    print('library nodes:', sorted({type(n).__name__ for n, _ in sdfg.all_nodes_recursive()
                                    if isinstance(n, dace.nodes.LibraryNode)}))
    out = np.zeros(2)
    sdfg(a=a, out=out, N=8)
    print('expected [2.0, -1.0], got', out.tolist())
    print('MISMATCH' if not np.allclose(out, [2.0, -1.0]) else 'ok')
