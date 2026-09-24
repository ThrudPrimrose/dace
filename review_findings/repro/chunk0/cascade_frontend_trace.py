"""Run canonicalize stages until the first CascadeInterstateEdgeAssignmentsUp application that moves something."""
from dace import symbolic
from dace.transformation.passes.canonicalize.pipeline import _build_stages
from dace.transformation.passes.canonicalize.cascade_iedge_assignments_up import CascadeInterstateEdgeAssignmentsUp
from cascade_frontend import prog


def dump(sdfg, tag):
    print(tag)
    for e in sdfg.all_interstate_edges():
        if e.data.assignments:
            print('   ', type(e.src).__name__, e.src.label, '->', type(e.dst).__name__, e.dst.label, e.data.assignments)


sdfg = prog.to_sdfg()
authority = {n: t for s in sdfg.all_sdfgs_recursive() for n, t in s.symbols.items()}
with symbolic.serialization_symbol_dtypes(authority):
    for i, (label, unit) in enumerate(_build_stages()):
        if isinstance(unit, CascadeInterstateEdgeAssignmentsUp):
            dump(sdfg, f'before stage {i} ({label})')
            r = unit.apply_pass(sdfg, {})
            if r:
                dump(sdfg, f'after: moved {r}')
                break
        else:
            unit.apply_pass(sdfg, {})
