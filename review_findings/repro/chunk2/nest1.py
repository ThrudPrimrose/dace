"""LoopToSymmetrize / LoopToTranspose visit nested-SDFG loops with the ROOT SDFG's descriptors."""
import sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize.loop_to_symmetrize import LoopToSymmetrize
from dace.transformation.passes.canonicalize.loop_to_transpose import LoopToTranspose
from harness import run_pass_checked, run_until
from dace.transformation.passes.canonicalize import canonicalize

M = dace.symbol('M')
K = dace.symbol('K')


@dace.program
def sym_inner(X: dace.float64[4, 4]):
    for i in range(0, 3):
        for j in range(i + 1, 4):
            X[j, i] = X[i, j]


@dace.program
def sym_outer(X: dace.float64[6, 6]):
    sym_inner(X[0:4, 0:4])


@dace.program
def tr_inner(A: dace.float64[4, 4], B: dace.float64[4, 4]):
    for i in range(4):
        for j in range(4):
            B[i, j] = A[j, i]


@dace.program
def tr_outer(A: dace.float64[6, 6], B: dace.float64[6, 6]):
    tr_inner(A[0:4, 0:4], B[0:4, 0:4])


mode, which = sys.argv[1], sys.argv[2]
prog, P = {'sym': (sym_outer, LoopToSymmetrize), 'tr': (tr_outer, LoopToTranspose)}[which]
sdfg = prog.to_sdfg(simplify=(mode not in ('direct', 'noop')))
sdfg.name = f'nest1_{which}_{mode}'
print('nested sdfgs:', len(list(sdfg.all_sdfgs_recursive())) - 1)
if mode == 'noop':
    for nsd in list(sdfg.all_sdfgs_recursive())[1:]:
        nsd.simplify()
elif mode == 'direct':
    for nsd in list(sdfg.all_sdfgs_recursive())[1:]:
        nsd.simplify()
    run_pass_checked(P(), sdfg)
elif mode == 'prefix':
    run_until(sdfg, P)
    print('nested sdfgs at pass:', len(list(sdfg.all_sdfgs_recursive())) - 1)
    run_pass_checked(P(), sdfg)
else:
    canonicalize(sdfg)
for sd in sdfg.all_sdfgs_recursive():
    print('  sdfg', sd.name, {k: (tuple(v.shape), type(v).__name__) for k, v in sd.arrays.items() if not k.startswith('__')})
for n_, st_ in sdfg.all_nodes_recursive():
    if isinstance(n_, dace.nodes.LibraryNode):
        print('  libnode', n_, [str(e.data) for e in st_.all_edges(n_)])
sdfg.validate()
m = 4
rng = np.random.default_rng(1)
if which == 'sym':
    X = rng.standard_normal((6, 6))
    ref = X.copy()
    for i in range(m - 1):
        for j in range(i + 1, m):
            ref[j, i] = ref[i, j]
    sdfg(X=X)
    print('ok:', np.allclose(X, ref))
else:
    A = rng.standard_normal((6, 6))
    B = np.zeros((6, 6))
    ref = B.copy()
    ref[:m, :m] = A[:m, :m].T
    sdfg(A=A, B=B)
    print('ok:', np.allclose(B, ref))
