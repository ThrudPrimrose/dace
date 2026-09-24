"""Frontend + full canonicalize: a loop-carried flag reset to an invariant value after its first use."""
import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')


@dace.program
def prog(A: dace.int64[N], K: dace.int64):
    m = 0
    for i in range(N):
        if m > 5:
            A[i] = 1
        else:
            A[i] = 2
        m = K + 1


if __name__ == '__main__':
    A0 = np.zeros(6, dtype=np.int64)
    ref = A0.copy()
    m = 0
    for i in range(6):
        ref[i] = 1 if m > 5 else 2
        m = 10 + 1
    sdfg = prog.to_sdfg()
    sdfg = canonicalize(sdfg)
    for e in sdfg.all_interstate_edges():
        if e.data.assignments:
            print('  edge', e.src.label, '->', e.dst.label, e.data.assignments)
    A = A0.copy()
    sdfg(A=A, K=10, N=6)
    print('numpy :', ref)
    print('dace  :', A)
    print('MISMATCH' if not np.array_equal(ref, A) else 'ok')
