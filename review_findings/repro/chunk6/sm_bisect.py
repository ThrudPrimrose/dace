import dace
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization import vectorize_multi_dim as vmd
s = dace.SDFG.from_file('ib_canon.sdfg')
v = VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=ISA.SCALAR, remainder_strategy='masked_tail'))
print([type(p).__name__ for p in v.passes])
import inspect
print(inspect.getsource(vmd.VectorizeMultiDim._vectorize_once)[:3000])
