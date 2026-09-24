"""Derived-symbol IV: ``k = i + 1`` in a loop that runs zero times must leave k at its entry value."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def derived(a: dace.float64[10], out: dace.int64[1], cnt: dace.int64[1]):
    k = cnt[0]
    for i in range(N):
        k = i + 1
        a[i] = k
    out[0] = k


def run(sdfg, n):
    a, out, cnt = np.zeros(10), np.zeros(1, np.int64), np.array([100], np.int64)
    sdfg(a=a, out=out, cnt=cnt, N=n)
    return out[0]


if __name__ == '__main__':
    sdfg = derived.to_sdfg(simplify=True)
    print('frontend only: N=0 out =', run(sdfg, 0), ' N=4 out =', run(sdfg, 4))
    sdfg = derived.to_sdfg(simplify=True)
    sdfg.name = 'derived_canon'
    canonicalize(sdfg, validate=True)
    print('canonicalized: N=0 out =', run(sdfg, 0), '(expected 100)  N=4 out =', run(sdfg, 4), '(expected 4)')
