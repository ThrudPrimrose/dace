"""RerollUnrolledLoops alone: overlapping lanes (step < m*g) on a zero-trip loop run extra iterations."""
import dace, numpy as np
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.reroll_unrolled_loops import RerollUnrolledLoops

N = dace.symbol('N')

@dace.program
def rr2i(a: dace.float64[N], b: dace.float64[N]):
    for i in range(0, N - 2, 2):
        a[i] = b[i]
        a[i + 1] = b[i + 1]
        a[i + 2] = b[i + 2]

def run(sdfg, n):
    a = np.zeros(n); b = np.arange(1, n + 1, dtype=np.float64)
    sdfg(a=a, b=b, N=n)
    return a

base = rr2i.to_sdfg(simplify=True)
before = {n: run(base, n) for n in (1, 2, 7)}
sdfg = rr2i.to_sdfg(simplify=True)
print('reroll returned', RerollUnrolledLoops().apply_pass(sdfg, {}))
for r in sdfg.all_control_flow_regions():
    if isinstance(r, LoopRegion):
        print('loop:', r.init_statement.as_string, '|', r.loop_condition.as_string, '|', r.update_statement.as_string)
sdfg.name = 'rr2i_after'
for n in (1, 2, 7):
    print(n, 'before', before[n], 'after', run(sdfg, n))
