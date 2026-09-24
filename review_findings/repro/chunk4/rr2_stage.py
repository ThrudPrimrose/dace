import dace, numpy as np
import isolate
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
N = dace.symbol('N')

@dace.program
def rr2s(a: dace.float64[N], b: dace.float64[N]):
    for i in range(0, N - 2, 2):
        a[i] = b[i]
        a[i + 1] = b[i + 1]
        a[i + 2] = b[i + 2]

sdfg = rr2s.to_sdfg(simplify=True)
canonicalize(sdfg)
print('reroll fired:', isolate.captured.get('ret'))
for tag in ('before', 'after'):
    s = isolate.captured[tag]; s.name = 'rr2s_' + tag
    for r in s.all_control_flow_regions(recursive=True):
        if isinstance(r, LoopRegion):
            print(tag, 'loop:', r.init_statement.as_string, '|', r.loop_condition.as_string, '|', r.update_statement.as_string)
    for n in (1, 2, 7):
        a = np.zeros(n); b = np.arange(1, n + 1, dtype=np.float64)
        s(a=a, b=b, N=n)
        print(tag, n, a)
