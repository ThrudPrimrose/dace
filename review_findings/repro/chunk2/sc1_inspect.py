import sys
sys.argv = ['x', 'none', 'ragged']
import dace
from dace.sdfg.state import LoopRegion
from harness import canon_prefix
N = dace.symbol('N'); M = dace.symbol('M')
@dace.program
def ragged(b: dace.float64[N, M], cnt: dace.int64[N], a: dace.float64[N * M], out: dace.int64[1]):
    j = -1
    for i in range(N):
        for k in range(cnt[i]):
            if b[i, k] > 0.0:
                j = j + 1
                a[j] = b[i, k]
    out[0] = j
sdfg = ragged.to_sdfg(simplify=True)
canon_prefix(sdfg, 'loop_to_x')
for r in sdfg.all_control_flow_regions():
    if isinstance(r, LoopRegion):
        print(r.label, r.loop_variable, r.init_statement.as_string, r.loop_condition.as_string)
        for e in r.edges():
            print('   edge', e.src.label, '->', e.dst.label, e.data.assignments)
