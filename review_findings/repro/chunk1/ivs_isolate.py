"""Walk the recipe up to the IV fixpoint, then apply InductionVariableSubstitution alone."""
import numpy as np
import dace
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize import pipeline as cp
from dace.transformation.passes.canonicalize.induction_variable_substitution import InductionVariableSubstitution
from ivs_prog import counted

sdfg = counted.to_sdfg(simplify=True)
sdfg.name = 'counted_iso2'
with dace.symbolic.serialization_symbol_dtypes({k: v for k, v in sdfg.symbols.items()}):
    for label, unit in cp._build_stages():
        if isinstance(unit, cp.IvSubstitutionFissionFixpoint):
            break
        unit.apply_pass(sdfg, {})
    print('loops before IVS:', sum(isinstance(b, LoopRegion) for b in sdfg.all_control_flow_blocks(recursive=True)))
    print('IVS result:', InductionVariableSubstitution().apply_pass(sdfg, {}))
    print('loops after IVS:', sum(isinstance(b, LoopRegion) for b in sdfg.all_control_flow_blocks(recursive=True)))
s, p = np.array([10.0]), np.array([10.0])
sdfg(s=s, p=p, M=5, N=3)
print('isolated: expected s=10 p=10, got s =', s[0], 'p =', p[0])
