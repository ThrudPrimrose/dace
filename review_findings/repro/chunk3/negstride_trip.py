import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.normalize_negative_stride import NormalizeNegativeStride

N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def prog(A: dace.float64[20]):
    for i in range(N, M, -2):
        A[i] = A[i] + 1.0

sdfg = prog.to_sdfg(simplify=True)
for cfg in sdfg.all_control_flow_regions():
    if isinstance(cfg, LoopRegion):
        print('before:', cfg.loop_variable, cfg.init_statement.as_string, cfg.loop_condition.as_string, cfg.update_statement.as_string)
print('pass returned', NormalizeNegativeStride().apply_pass(sdfg, {}))
for cfg in sdfg.all_control_flow_regions():
    if isinstance(cfg, LoopRegion):
        print('after:', cfg.loop_variable, cfg.init_statement.as_string, cfg.loop_condition.as_string, cfg.update_statement.as_string)
sdfg.validate()
A = np.zeros(20)
ref = np.zeros(20)
for i in range(5, 5, -2):
    ref[i] += 1
sdfg(A=A, N=5, M=5)
print('N=5,M=5 dace:', A.nonzero()[0].tolist(), 'numpy:', ref.nonzero()[0].tolist())
A = np.zeros(20); ref = np.zeros(20)
for i in range(5, 8, -2):
    ref[i] += 1
sdfg(A=A, N=5, M=8)
print('N=5,M=8 dace:', A.nonzero()[0].tolist(), 'numpy:', ref.nonzero()[0].tolist())
