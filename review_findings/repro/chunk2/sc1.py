import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize.loop_to_stream_compaction import LoopToStreamCompaction
from harness import run_pass_checked, canon_prefix
from dace.transformation.passes.canonicalize import canonicalize

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def ragged(b: dace.float64[N, M], cnt: dace.int64[N], a: dace.float64[N * M], out: dace.int64[1]):
    j = -1
    for i in range(N):
        for k in range(cnt[i]):
            if b[i, k] > 0.0:
                j = j + 1
                a[j] = b[i, k]
    out[0] = j


@dace.program
def carried(b: dace.int64[N], a: dace.int64[N], out: dace.int64[1]):
    j = -1
    last = -1000
    for i in range(N):
        if b[i] > last:
            j = j + 1
            a[j] = b[i]
            last = b[i]
    out[0] = j


def ref_ragged(b, cnt):
    a = np.zeros(b.size)
    j = -1
    for i in range(b.shape[0]):
        for k in range(cnt[i]):
            if b[i, k] > 0:
                j += 1
                a[j] = b[i, k]
    return a, j


def ref_carried(b):
    a = np.zeros(b.size, dtype=np.int64)
    j = -1
    last = -1000
    for x in b:
        if x > last:
            j += 1
            a[j] = x
            last = x
    return a, j


mode, which = sys.argv[1], sys.argv[2]
prog = {'ragged': ragged, 'carried': carried}[which]
sdfg = prog.to_sdfg(simplify=True)
sdfg.name = f'sc1_{which}_{mode}'
if mode == 'direct':
    run_pass_checked(LoopToStreamCompaction(), sdfg)
elif mode == 'prefix':
    canon_prefix(sdfg, 'loop_to_x')
    run_pass_checked(LoopToStreamCompaction(), sdfg)
else:
    canonicalize(sdfg)
sdfg.validate()
print('has Scan:', any(type(n).__name__ == 'Scan' for n, _ in sdfg.all_nodes_recursive()))
out = np.zeros(1, dtype=np.int64)
if which == 'ragged':
    n, m = 5, 6
    rng = np.random.default_rng(0)
    b = rng.standard_normal((n, m))
    cnt = np.array([6, 2, 0, 4, 3], dtype=np.int64)
    a = np.zeros(n * m)
    sdfg(b=b, cnt=cnt, a=a, out=out, N=n, M=m)
    ra, rj = ref_ragged(b, cnt)
else:
    n = 12
    b = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 10], dtype=np.int64)
    a = np.zeros(n, dtype=np.int64)
    sdfg(b=b, a=a, out=out, N=n)
    ra, rj = ref_carried(b)
print('j got', out[0], 'expected', rj, '; a ok:', np.allclose(a, ra))
print(a)
print(ra)
