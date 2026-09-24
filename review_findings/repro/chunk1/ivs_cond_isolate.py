"""Walk the recipe up to the IV fixpoint, then run InductionVariableSubstitution alone."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize import pipeline as cp
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.simplify import SimplifyPass
from ivs_cond_increment import cond_counter, reference, run

a = np.array([0.9, 0.1, 0.7, 0.2, 0.8, 0.3])
rb, rk = reference(a)
for apply_ivs in (False, True):
    sdfg = cond_counter.to_sdfg(simplify=True)
    sdfg.name = f'cond_iso_{int(apply_ivs)}'
    with dace.symbolic.serialization_symbol_dtypes(dict(sdfg.symbols)):
        for label, unit in cp._build_stages():
            if isinstance(unit, cp.IvSubstitutionFissionFixpoint):
                break
            unit.apply_pass(sdfg, {})
        promote = ScalarToSymbolPromotion()
        promote.transients_only = False
        promote.apply_pass(sdfg, {})
        SimplifyPass().apply_pass(sdfg, {})
        if apply_ivs:
            print('IVS result:', InductionVariableSubstitution().apply_pass(sdfg, {}))
            for e in sdfg.all_interstate_edges():
                if e.data.assignments:
                    print('   assignments', e.data.assignments)
    gb, gk = run(sdfg, a)
    print('IVS applied' if apply_ivs else 'no IVS     ', 'b ok', np.array_equal(gb, rb), 'k', gk, 'expected', rk)
