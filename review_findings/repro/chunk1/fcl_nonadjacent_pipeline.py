"""Same shape as fcl_nonadjacent.py, through the full canonicalize() pipeline."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize

N, M, K = (dace.symbol(s, dtype=dace.int64) for s in 'NMK')


@dace.program
def split_sum_pipe(a: dace.float64[N], acc: dace.float64[1]):
    for i in range(0, M):
        acc[0] = acc[0] + a[i]
    for j in range(M, K):
        acc[0] = acc[0] + a[j]


a = np.arange(1.0, 11.0)
sdfg = split_sum_pipe.to_sdfg(simplify=True)
canonicalize(sdfg, validate=True)
for mk in ((5, 3), (3, 5)):
    acc = np.zeros(1)
    sdfg(a=a, acc=acc, N=10, M=mk[0], K=mk[1])
    ref = a[0:mk[0]].sum() + a[mk[0]:mk[1]].sum()
    print('M,K =', mk, 'reference', ref, 'got', acc[0])
