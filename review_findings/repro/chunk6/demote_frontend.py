import sys
import numpy as np
import dace
from drv import run

N = dace.symbol('N')
M = dace.symbol('M')

@dace.program
def inner_bound(cnt: dace.int32[1], A: dace.float64[M, N], out: dace.float64[M]):
    n = cnt[0]
    for i in dace.map[0:M]:
        s = 0.0
        for k in range(n):
            s = s + A[i, k]
        out[i] = s

@dace.program
def inner_bound2(cnt: dace.int32[1], A: dace.float64[M, N], out: dace.float64[M, N]):
    n = cnt[0]
    for i in dace.map[0:M]:
        for k in dace.map[0:n]:
            out[i, k] = A[i, k] * 2.0

m, nn = 5, 12
rng = np.random.default_rng(0)
for prog, arrs in ((inner_bound, dict(cnt=np.array([7], np.int32), A=rng.random((m, nn)), out=np.zeros(m))),
                   (inner_bound2, dict(cnt=np.array([7], np.int32), A=rng.random((m, nn)), out=np.zeros((m, nn))))):
    print('==', prog.name)
    try:
        run(prog, prog.name + '_d', arrs, dict(N=nn, M=m), remainder='masked_tail', exact=False)
    except Exception as e:
        print('EXCEPTION', type(e).__name__, str(e)[:300])
