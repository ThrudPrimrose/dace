"""recognize_map_reduction (utils/reductions.py) treats an element-wise in-place update
``y[j] = y[j] + x[cols[j]]`` as a loop-carried scalar reduction because it only checks that the
accumulator memlets are single-element, not that they are the SAME element for every iteration.
LiftMapReductionToReduce(rmw_only=True) then folds all of x into y[0]."""
import copy
import numpy as np
import dace
from drv import vec, ntile
from dace.transformation.passes.vectorization.utils.reductions import recognize_map_reduction

N = 16


@dace.program
def upd(x: dace.float64[N], cols: dace.int32[N], y: dace.float64[N]):
    y[0] = 0.0
    for j in dace.map[0:N]:
        y[j] = y[j] + x[cols[j]]


def run(sdfg):
    x = np.arange(1, N + 1, dtype=np.float64)
    cols = np.arange(N, dtype=np.int32)[::-1].copy()
    expected = np.full(N, 100.0)
    expected[0] = 0.0
    expected += x[cols]
    y = np.full(N, 100.0)
    sdfg(x=x, cols=cols, y=y)
    return y, expected


if __name__ == '__main__':
    s = upd.to_sdfg(simplify=True)
    for n, g in s.all_nodes_recursive():
        if isinstance(n, dace.nodes.MapEntry):
            info = recognize_map_reduction(g, n)
            print('recognize_map_reduction ->',
                  None if info is None else (info.op, info.accumulator, str(info.read_edge.data), str(info.write_edge.data)))
    ref = copy.deepcopy(s)
    ref.name = 'upd_ref'
    y, expected = run(ref)
    print('unvectorized ok', np.allclose(y, expected))
    vec(s)
    print('Reduce nodes', sum(isinstance(n, dace.libraries.standard.Reduce) for n, _ in s.all_nodes_recursive()))
    y, expected = run(s)
    print('vectorized   ok', np.allclose(y, expected))
    print('got     ', y)
    print('expected', expected)
