"""Probe FuseLoops legality on sequential sibling loops."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize.fuse_loops import FuseLoops

N = dace.symbol('N', dtype=dace.int64)


@dace.program
def fl_last(a: dace.float64[N], c: dace.float64[N]):
    for i in range(1, N):
        a[i] = a[i - 1] + 1.0
    for j in range(1, N):
        c[j] = c[j - 1] + a[N - 1]


@dace.program
def fl_scalar(a: dace.float64[N], c: dace.float64[N], s: dace.float64[1]):
    for i in range(1, N):
        a[i] = a[i - 1] + s[0]
        s[0] = a[i]
    for j in range(1, N):
        c[j] = c[j - 1] + s[0]


@dace.program
def fl_first(a: dace.float64[N], c: dace.float64[N]):
    for i in range(1, N):
        c[i] = c[i - 1] + a[1]
    for j in range(1, N):
        a[j] = a[j - 1] + 1.0


for prog in (fl_last, fl_scalar, fl_first):
    sdfg = prog.to_sdfg(simplify=True)
    ref_sdfg = prog.to_sdfg(simplify=True)
    ref_sdfg.name = prog.name + '_ref'
    res = FuseLoops().apply_pass(sdfg, {})
    print(prog.name, 'FuseLoops result:', res)
    if not res:
        continue
    rng = np.random.default_rng(0)
    args = {k: rng.random(v.shape if not dace.symbolic.issymbolic(v.shape[0]) else (8, )) for k, v in sdfg.arrays.items()
            if not v.transient}
    ref_args = {k: v.copy() for k, v in args.items()}
    sdfg(**args, N=8)
    ref_sdfg(**ref_args, N=8)
    print('   max diff', max(np.abs(args[k] - ref_args[k]).max() for k in args))
