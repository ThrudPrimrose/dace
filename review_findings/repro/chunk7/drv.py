"""Tiny driver: to_sdfg -> simplify -> canonicalize -> VectorizeCPUMultiDim -> compile -> compare."""
import copy
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim


def vectorize(prog, name, widths=(8, ), remainder="scalar_postamble", branch_mode="merge", sdfg=None, canon=True, **kw):
    if sdfg is None:
        sdfg = prog.to_sdfg(simplify=False)
        sdfg.simplify()
    sdfg.name = name
    if canon:
        canonicalize(sdfg, validate=True)
    VectorizeCPUMultiDim(VectorizeConfig(widths=widths, target_isa=ISA.SCALAR, remainder_strategy=remainder,
                                         branch_mode=branch_mode, **kw)).apply_pass(sdfg, {})
    sdfg.validate()
    return sdfg


def run(sdfg, args, **syms):
    a = {k: copy.deepcopy(v) for k, v in args.items()}
    sdfg(**a, **syms)
    return a


def compare(ref, got):
    ok = True
    for k in ref:
        if isinstance(ref[k], np.ndarray) and not np.allclose(ref[k], got[k]):
            print("MISMATCH", k, "\n ref", ref[k].ravel()[:20], "\n got", got[k].ravel()[:20])
            ok = False
    print("MATCH" if ok else "FAIL")
    return ok
