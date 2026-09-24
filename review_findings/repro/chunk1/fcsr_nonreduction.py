"""FuseChainedScalarReductions checks only the TOP-LEVEL operator of a tasklet, so
``o = acc * 0.5 + inc`` (a linear recurrence, not a reduction) is re-associated."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.canonicalize.fuse_chained_scalar_reductions import FuseChainedScalarReductions

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def damped(a: dace.float64[N], b: dace.float64[N], s: dace.float64[1]):
    for i in range(N):
        with dace.tasklet:
            x << s[0]
            y << a[i]
            o >> s[0]
            o = x * 0.5 + y
        with dace.tasklet:
            x2 << s[0]
            y2 << b[i]
            o2 >> s[0]
            o2 = x2 * 0.5 + y2


def reference(a, b):
    s = 0.0
    for i in range(len(a)):
        s = s * 0.5 + a[i]
        s = s * 0.5 + b[i]
    return s


rng = np.random.default_rng(0)
a, b = rng.random(8), rng.random(8)

# Direct pass application on the frontend SDFG.
sdfg = damped.to_sdfg(simplify=True)
print('direct pass result:', FuseChainedScalarReductions().apply_pass(sdfg, {}))
s = np.zeros(1)
sdfg(a=a, b=b, s=s, N=8)
print('direct:   reference', reference(a, b), 'got', s[0])

# Full pipeline.
sdfg = damped.to_sdfg(simplify=True)
canonicalize(sdfg, validate=True)
s = np.zeros(1)
sdfg(a=a, b=b, s=s, N=8)
print('pipeline: reference', reference(a, b), 'got', s[0])
