"""FuseConsecutiveLoops fuses [0, M) and [M, K) into [0, K) without proving M <= K."""
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.fuse_consecutive_loops import FuseConsecutiveLoops

N, M, K = (dace.symbol(s, dtype=dace.int64) for s in 'NMK')


@dace.program
def split_sum(a: dace.float64[N], acc: dace.float64[1]):
    for i in range(0, M):
        acc[0] = acc[0] + a[i]
    for j in range(M, K):
        acc[0] = acc[0] + a[j]


a = np.arange(1.0, 11.0)
ref = a[0:5].sum()  # M=5, K=3: the second loop is empty
sdfg = split_sum.to_sdfg(simplify=True)
print('loops before:', sum(isinstance(b, LoopRegion) for b in sdfg.all_control_flow_blocks(recursive=True)))
print('pass result:', FuseConsecutiveLoops().apply_pass(sdfg, {}))
for b in sdfg.all_control_flow_blocks(recursive=True):
    if isinstance(b, LoopRegion):
        print('  loop', b.loop_variable, b.init_statement.as_string, '|', b.loop_condition.as_string)
acc = np.zeros(1)
sdfg(a=a, acc=acc, N=10, M=5, K=3)
print('reference', ref, 'got', acc[0])
