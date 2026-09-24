"""LoopToSymm (map form) lifts a symm nest whose map covers only part of C to a full-array Symm node."""
import sys
import numpy as np
import dace
from dace.libraries.blas.nodes.symm import Symm
from dace.transformation.passes.canonicalize.loop_to_symm import LoopToSymm
from dace.transformation.passes.canonicalize import canonicalize
from harness import run_pass_checked

M = dace.symbol("M")
N = dace.symbol("N")
datatype = dace.float64


@dace.program
def symm_partial(C: datatype[M, N], A: datatype[M, M], B: datatype[M, N], alpha: datatype[1], beta: datatype[1]):

    @dace.mapscope
    def comp_all(j: _[0:N - 1], i: _[0:M]):  # last column of C is NOT touched
        temp2 = dace.define_local_scalar(datatype)

        @dace.tasklet
        def reset_tmp():
            tmp >> temp2
            tmp = 0

        @dace.map
        def comp_t2(k: _[0:i]):
            ialpha << alpha
            ia << A[i, k]
            ibi << B[i, j]
            ibk << B[k, j]
            oc >> C(1, lambda a, b: a + b)[k, j]
            ot2 >> temp2(1, lambda a, b: a + b)

            oc = ialpha * ibi * ia
            ot2 = ibk * ia

        @dace.tasklet
        def comp_rest():
            ibeta << beta
            ib << B[i, j]
            iadiag << A[i, i]
            ialpha << alpha
            it2 << temp2
            ic << C[i, j]
            oc >> C[i, j]
            oc = ibeta * ic + ialpha * ib * iadiag + ialpha * it2


mode = sys.argv[1]
sdfg = symm_partial.to_sdfg(simplify=False)
sdfg.name = f'symm1_{mode}'
if mode == 'direct':
    run_pass_checked(LoopToSymm(), sdfg)
else:
    canonicalize(sdfg, validate=True)
print('Symm nodes:', sum(isinstance(n, Symm) for n, _ in sdfg.all_nodes_recursive()))
m, n = 5, 4
rng = np.random.default_rng(0)
A = np.tril(rng.random((m, m)))
B = rng.random((m, n))
C = rng.random((m, n))
alpha, beta = np.array([1.5]), np.array([1.2])
Asym = np.tril(A) + np.tril(A, -1).T
ref = C.copy()
ref[:, :n - 1] = alpha[0] * (Asym @ B)[:, :n - 1] + beta[0] * C[:, :n - 1]
Cw = C.copy()
sdfg(C=Cw, A=A.copy(), B=B.copy(), alpha=alpha, beta=beta, M=m, N=n)
print('ok:', np.allclose(Cw, ref), '; last column untouched:', np.allclose(Cw[:, -1], C[:, -1]))
