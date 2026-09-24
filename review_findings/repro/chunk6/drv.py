"""Shared driver: reference compile vs canonicalize + VectorizeCPUMultiDim (as the harness does)."""
import copy
import warnings
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.libraries import tileops


def count_tile_nodes(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if type(n).__module__.startswith('dace.libraries.tileops'))


def run(prog, name, arrays, params, widths=(8, ), remainder='scalar_postamble', fma=False, sdfg=None, exact=True,
        reference=None, lift_copy=True):
    ref_sdfg = prog.to_sdfg(simplify=True) if sdfg is None else sdfg
    ref_sdfg.name = name + '_ref'
    vec = copy.deepcopy(ref_sdfg)
    vec.name = name + '_vec'
    a_ref = {k: copy.deepcopy(v) for k, v in arrays.items()}
    a_vec = {k: copy.deepcopy(v) for k, v in arrays.items()}
    if reference is None:
        ref_sdfg(**a_ref, **params)
    else:
        reference(a_ref)
    canonicalize(vec, validate=True, lift_copy=lift_copy)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        VectorizeCPUMultiDim(
            VectorizeConfig(widths=widths, target_isa=ISA.SCALAR, remainder_strategy=remainder,
                            fuse_multiply_add=fma)).apply_pass(vec, {})
    for x in w:
        if 'VectorizeMultiDim' in str(x.message):
            print('WARN:', str(x.message)[:300])
    print('tile nodes emitted:', count_tile_nodes(vec))
    vec.validate()
    vec(**a_vec, **params)
    ok = True
    for k in arrays:
        same = np.array_equal(a_ref[k], a_vec[k]) if exact else np.allclose(a_ref[k], a_vec[k], rtol=1e-12, atol=0)
        if not same:
            ok = False
            diff = np.nonzero(a_ref[k] != a_vec[k])
            print(f'MISMATCH {k}: first bad idx {tuple(d[:5] for d in diff)} ref {a_ref[k][diff][:5]} vec {a_vec[k][diff][:5]}')
    print('RESULT:', 'OK' if ok else 'WRONG')
    return vec
