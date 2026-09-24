"""Counter IV stepped on a top-level iedge AND inside a one-sided conditional in the same body."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def cond_counter(a: dace.float64[N], b: dace.float64[N], out: dace.int64[1]):
    k = 0
    for i in range(N):
        k = k + 1
        if a[i] > 0.5:
            k = k + 10
        b[i] = k
    out[0] = k


def reference(a):
    k, b = 0, np.zeros_like(a)
    for i in range(len(a)):
        k += 1
        if a[i] > 0.5:
            k += 10
        b[i] = k
    return b, k


def run(sdfg, a):
    b, out = np.zeros_like(a), np.zeros(1, np.int64)
    sdfg(a=a, b=b, out=out, N=len(a))
    return b, out[0]


if __name__ == '__main__':
    a = np.array([0.9, 0.1, 0.7, 0.2, 0.8, 0.3])
    rb, rk = reference(a)
    sdfg = cond_counter.to_sdfg(simplify=True)
    gb, gk = run(sdfg, a)
    print('frontend only: b ok', np.array_equal(gb, rb), 'k', gk, 'expected', rk)
    sdfg = cond_counter.to_sdfg(simplify=True)
    sdfg.name = 'cond_counter_canon'
    canonicalize(sdfg, validate=True)
    gb, gk = run(sdfg, a)
    print('canonicalized: b', gb, 'expected', rb, '| k', gk, 'expected', rk)
