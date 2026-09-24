import copy, warnings, sys
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.libraries.tileops.nodes import TileLoad, TileStore

def tiles(sdfg):
    from dace.transformation.passes.vectorization.vectorize_multi_dim import EMITTABLE_TILE_NODE_TYPES
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, EMITTABLE_TILE_NODE_TYPES)]

def run(prog, args, widths=(8,), remainder="masked_tail", name=None, show=False, expand=False, **kw):
    sdfg = prog.to_sdfg(simplify=True)
    sdfg.name = (name or prog.name)
    ref = {k: copy.deepcopy(v) for k, v in args.items()}
    vec = {k: copy.deepcopy(v) for k, v in args.items()}
    sdfg_ref = copy.deepcopy(sdfg); sdfg_ref.name += "_ref"
    canonicalize(sdfg, validate=True)
    cfg = VectorizeConfig(widths=widths, target_isa="SCALAR", remainder_strategy=remainder, expand_tile_nodes=expand, **kw)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        VectorizeCPUMultiDim(cfg).apply_pass(sdfg, {})
        for x in w:
            if 'Vectorize' in str(x.message): print("WARN:", str(x.message)[:300])
    t = tiles(sdfg)
    print("tile nodes:", len(t), sorted({type(n).__name__ for n in t}))
    if show:
        for n in t: print("  ", type(n).__name__, {p: getattr(n, p, None) for p in ('dim_strides','gather_dims','replicate_factors','op') if hasattr(n, p)})
    sdfg.name += "_vec"
    sdfg_ref(**ref)
    sdfg(**vec)
    ok = True
    for k in args:
        if isinstance(args[k], np.ndarray):
            if not np.allclose(ref[k], vec[k], equal_nan=True):
                ok = False
                print("MISMATCH", k); print(" ref", ref[k].ravel()[:24]); print(" vec", vec[k].ravel()[:24])
    print("OK" if ok else "FAIL")
    return sdfg
