"""Same as cascade_frontend.py, with CascadeInterstateEdgeAssignmentsUp disabled: isolates the culprit."""
import numpy as np
from dace.transformation.passes.canonicalize import cascade_iedge_assignments_up as cia
cia.CascadeInterstateEdgeAssignmentsUp.apply_pass = lambda self, sdfg, res: None
from dace.transformation.passes.canonicalize import canonicalize
from cascade_frontend import prog

sdfg = canonicalize(prog.to_sdfg())
A = np.zeros(6, dtype=np.int64)
sdfg(A=A, K=10, N=6)
print('dace (cascade disabled):', A)
