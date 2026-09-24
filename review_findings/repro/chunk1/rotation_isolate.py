"""Run the recipe up to the 'rotate' stage, then LoopCarriedRotationSubstitution alone."""
import dace
from dace.transformation.passes.canonicalize import pipeline as cp
from dace.transformation.passes.canonicalize.induction_variable_substitution import LoopCarriedRotationSubstitution
from rotation_zero_trip import delay_line, run

for apply_rot in (False, True):
    sdfg = delay_line.to_sdfg(simplify=True)
    sdfg.name = f'delay_iso_{int(apply_rot)}'
    with dace.symbolic.serialization_symbol_dtypes(dict(sdfg.symbols)):
        for label, unit in cp._build_stages():
            if label == 'rotate':
                break
            unit.apply_pass(sdfg, {})
        if apply_rot:
            print('LoopCarriedRotationSubstitution result:', LoopCarriedRotationSubstitution(4).apply_pass(sdfg, {}))
    got, ref = run(sdfg, 0)
    print('rotation applied' if apply_rot else 'no rotation     ', 'N=0 a[:3] =', got[:3], 'expected', ref[:3])
