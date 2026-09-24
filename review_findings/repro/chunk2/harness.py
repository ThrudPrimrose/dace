"""Shared helpers for chunk2 reproducers."""
import copy
import json

import dace
from dace.transformation.passes.canonicalize import canonicalize, stage_labels
from dace.transformation.passes.canonicalize.pipeline import CanonicalizationPipeline


def stages_before(label: str) -> list[str]:
    labels = stage_labels()
    return labels[:labels.index(label)]


def stages_from(label: str) -> list[str]:
    labels = stage_labels()
    return labels[labels.index(label):]


def canon_prefix(sdfg: dace.SDFG, label: str) -> dace.SDFG:
    """Run every stage before ``label``."""
    canonicalize(sdfg, stages=stages_before(label))
    return sdfg


def digest(sdfg: dace.SDFG) -> str:
    j = sdfg.to_json()

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ('guid', 'cfg_list_id', 'hash')}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o

    return json.dumps(strip(j), sort_keys=True)


def run_pass_checked(p, sdfg: dace.SDFG):
    before = digest(sdfg)
    res = p.apply_pass(sdfg, {})
    after = digest(sdfg)
    print(f'{type(p).__name__}: result={res} changed={before != after}')
    return res


def run_until(sdfg: dace.SDFG, pass_type: type, occurrence: int = 0, **kw) -> dace.SDFG:
    """Run the canonicalize recipe unit by unit, stopping right before the ``occurrence``-th unit of
    ``pass_type`` (so the caller can apply that pass itself, on exactly the graph the pipeline hands it)."""
    from dace import symbolic
    from dace.transformation.passes.canonicalize import pipeline as pl
    authority = {name: dtype for nested in sdfg.all_sdfgs_recursive() for name, dtype in nested.symbols.items()}
    with symbolic.serialization_symbol_dtypes(authority):
        pipe = CanonicalizationPipeline(**kw)
        pl.disable_openmp_sections(sdfg)
        seen = 0
        for label, unit in pipe.build_stages():
            if isinstance(unit, pass_type):
                if seen == occurrence:
                    return sdfg
                seen += 1
            unit.apply_pass(sdfg, {})
    raise RuntimeError('pass not found in recipe')
