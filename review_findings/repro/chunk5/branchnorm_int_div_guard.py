# BranchNormalization lowers `if b[i] != 0: a[i] = c[i] // b[i]` to an unconditional compute of
# c[i] // b[i] followed by ITE(cond, new, old). A data guard that protects a trapping operation
# (integer division by zero) is thereby removed -> SIGFPE on any lane with b[i] == 0.
import subprocess, sys
import numpy as np
import dace
from dace.transformation.passes.parallelize import parallelize
from dace.transformation.passes.vectorization.branch_normalization import BranchNormalization
from dace.transformation.passes.vectorization.same_write_set_if_else_to_ite_cfg import SameWriteSetIfElseToITECFG

N = dace.symbol('N')


@dace.program
def guarded_div(a: dace.int64[N], b: dace.int64[N], c: dace.int64[N]):
    for i in range(N):
        if b[i] != 0:
            a[i] = c[i] // b[i]


def run(lower: bool):
    sdfg = guarded_div.to_sdfg(simplify=True)
    sdfg.name = f'guarded_div_{int(lower)}'
    parallelize(sdfg, validate=True, validate_all=False)
    if lower:
        print('SameWriteSet returned', SameWriteSetIfElseToITECFG().apply_pass(sdfg, {}))
        print('BranchNormalization returned', BranchNormalization().apply_pass(sdfg, {}))
        for sd in sdfg.all_sdfgs_recursive():
            for st in sd.states():
                for n in st.nodes():
                    if isinstance(n, dace.nodes.Tasklet):
                        print('  tasklet', n.label, ':', n.code.as_string.strip())
    sdfg.validate()
    a = np.full(8, -1, dtype=np.int64)
    b = np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype=np.int64)
    c = np.arange(8, dtype=np.int64) * 10
    sdfg(a=a, b=b, c=c, N=8)
    print('a =', a)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(sys.argv[1] == '1')
    else:
        for lower in ('0', '1'):
            p = subprocess.run([sys.executable, __file__, lower], capture_output=True, text=True)
            out = '\n'.join(l for l in (p.stdout + p.stderr).splitlines() if 'arn' not in l)
            print(f'--- lower={lower}: returncode {p.returncode}\n{out[-1500:]}')
