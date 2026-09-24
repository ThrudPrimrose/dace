import json, warnings, difflib
import numpy as np
import dace
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
N = dace.symbol('N'); M = dace.symbol('M')
@dace.program
def k_sum(a: dace.float64[N + M], b: dace.float64[N, M]):
    for i, j in dace.map[0:N, 0:M]:
        b[i, j] = a[i + j]
sdfg = k_sum.to_sdfg(simplify=True)
canonicalize(sdfg, validate=True)
before = json.dumps(sdfg.to_json(), sort_keys=True, indent=1)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r = VectorizeCPUMultiDim(VectorizeConfig(widths=(4, 4), target_isa="SCALAR")).apply_pass(sdfg, {})
    print([str(x.message)[:80] for x in w if 'Vectorize' in str(x.message)])
after = json.dumps(sdfg.to_json(), sort_keys=True, indent=1)
print("returned", r, "identical:", before == after)
for l in list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm='', n=1))[:40]: print(l)
import re
strip = lambda s: re.sub(r'"guid": "[^"]*"', '"guid": X', s)
print("identical modulo guids:", strip(before) == strip(after))
for l in list(difflib.unified_diff(strip(before).splitlines(), strip(after).splitlines(), lineterm='', n=2))[:40]: print(l)
