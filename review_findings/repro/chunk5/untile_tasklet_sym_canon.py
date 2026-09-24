# Same as untile_tasklet_sym.py, through the full canonicalize() pipeline.
import numpy as np
import dace
from dace.transformation.passes.canonicalize.pipeline import canonicalize

N = 16

@dace.program
def k(a: dace.float64[N]):
    for i in range(0, N, 4):
        for ii in range(0, 4):
            a[i + ii] = i

sdfg = k.to_sdfg(simplify=True)
ref = np.zeros(N); k.f(ref)
canonicalize(sdfg, validate=True)
a = np.zeros(N); sdfg(a=a)
print('ref', ref); print('got', a); print('match', np.allclose(a, ref))
