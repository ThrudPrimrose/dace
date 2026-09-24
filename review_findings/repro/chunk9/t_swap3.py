import sys, copy, warnings
sys.path.insert(0, '.')
import numpy as np, dace
from swapbuild2 import build
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
n = 16
a = np.random.rand(n, n)
ref = build(); ref.name += '_r'; b_ref = np.zeros((n, n)); ref(a=a, b=b_ref, N=n)
s = build()
try:
    VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa="SCALAR")).apply_pass(s, {})
    print("vectorize ok")
    b = np.zeros((n, n)); s(a=a, b=b, N=n); print("match", np.allclose(b, b_ref))
except Exception as ex:
    print("EXC", type(ex).__name__, str(ex)[:300])
