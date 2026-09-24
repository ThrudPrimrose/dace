"""Walk the recipe up to the IV fixpoint, then run InductionVariableSubstitution alone on the counter IV."""
import numpy as np
import dace
from dace.transformation.passes.canonicalize import pipeline as cp
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution
from dace.transformation.passes.scalar_to_symbol import ScalarToSymbolPromotion
from dace.transformation.passes.simplify import SimplifyPass
from ivs_negative_trip_iedge import counter, run

for apply_ivs in (False, True):
    sdfg = counter.to_sdfg(simplify=True)
    sdfg.name = f'counter_iso_{int(apply_ivs)}'
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
    print('IVS applied' if apply_ivs else 'no IVS     ', 'out =', run(sdfg)[0], '(expected 100)')
