"""Counter IV (``k += 1`` next to other work): exit value k + (N - M) is wrong when M > N."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize

M, N = (dace.symbol(s, dtype=dace.int64) for s in 'MN')


@dace.program
def counter(a: dace.float64[10], out: dace.int64[1], cnt: dace.int64[1]):
    k = cnt[0]
    for i in range(M, N):
        k = k + 1
        a[i] = k
    out[0] = k


def run(sdfg):
    a, out, cnt = np.zeros(10), np.zeros(1, np.int64), np.array([100], np.int64)
    sdfg(a=a, out=out, cnt=cnt, M=5, N=3)
    return out[0], a


if __name__ == '__main__':
    sdfg = counter.to_sdfg(simplify=True)
    print('frontend only: out =', run(sdfg)[0])
    sdfg = counter.to_sdfg(simplify=True)
    sdfg.name = 'counter_canon'
    canonicalize(sdfg, validate=True)
    print('canonicalized: out =', run(sdfg)[0], '(expected 100)')
