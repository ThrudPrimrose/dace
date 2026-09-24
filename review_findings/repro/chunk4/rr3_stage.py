"""RerollUnrolledLoops' new bound uses truncating int_floor: a zero-trip exact-coverage loop runs."""
import dace, numpy as np
import isolate
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import canonicalize
N = dace.symbol('N')

@dace.program
def rr3s(a: dace.float64[N + 2], b: dace.float64[N + 2]):
    for i in range(0, N - 1, 2):
        a[i] = b[i]
        a[i + 1] = b[i + 1]

sdfg = rr3s.to_sdfg(simplify=True)
final = sdfg
canonicalize(sdfg)
print('reroll fired:', isolate.captured.get('ret'))
for tag in ('before', 'after'):
    s = isolate.captured[tag]; s.name = 'rr3s_' + tag
    for r in s.all_control_flow_regions(recursive=True):
        if isinstance(r, LoopRegion):
            print(tag, 'loop:', r.init_statement.as_string, '|', r.loop_condition.as_string, '|', r.update_statement.as_string)
    for n in (0, 1, 5):
        a = np.zeros(n + 2); b = np.arange(1, n + 3, dtype=np.float64)
        s(a=a, b=b, N=n)
        print(tag, 'N=%d' % n, a)
for n in (0, 1, 5):
    a = np.zeros(n + 2); b = np.arange(1, n + 3, dtype=np.float64)
    final(a=a, b=b, N=n)
    print('canonicalized N=%d' % n, a)
