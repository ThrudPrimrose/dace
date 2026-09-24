import dace
from dace.transformation.passes.canonicalize import pipeline as cp
from ivs_cond_increment import cond_counter

sdfg = cond_counter.to_sdfg(simplify=True)
prev = None
with dace.symbolic.serialization_symbol_dtypes(dict(sdfg.symbols)):
    for idx, (label, unit) in enumerate(cp._build_stages()):
        unit.apply_pass(sdfg, {})
        cur = sorted(str(e.data.assignments) for e in sdfg.all_interstate_edges(recursive=True) if 'k' in e.data.assignments)
        if cur != prev:
            print(idx, label, type(unit).__name__, cur)
        prev = cur
