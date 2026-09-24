"""HoistLoopRangeCalls skips maps inside nested SDFGs (sdfg.all_states() does not recurse into NestedSDFG)."""
import numpy as np
import dace
from dace import subsets, symbolic
from dace.transformation.passes.canonicalize.hoist_loop_range_calls import HoistLoopRangeCalls, contains_call


def inner_sdfg() -> tuple[dace.SDFG, object]:
    sdfg = dace.SDFG('hoist_nested_inner')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', (128, ), dace.float64)
    state = sdfg.add_state()
    step = symbolic.pystr_to_symbolic('int_ceil(N, 4)')
    entry, exit_node = state.add_map('chunk', {'c': subsets.Range([(0, symbolic.pystr_to_symbolic('N') - 1, step)])},
                                     schedule=dace.ScheduleType.CPU_Multicore)
    tasklet = state.add_tasklet('w', {}, {'o'}, 'o = 1.0')
    write = state.add_write('A')
    state.add_memlet_path(entry, tasklet, memlet=dace.Memlet())
    state.add_memlet_path(tasklet, exit_node, write, src_conn='o', memlet=dace.Memlet('A[c]'))
    return sdfg, entry


def outer_sdfg() -> tuple[dace.SDFG, object]:
    sdfg = dace.SDFG('hoist_nested_outer')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', (128, ), dace.float64)
    state = sdfg.add_state()
    inner, entry = inner_sdfg()
    nsdfg = state.add_nested_sdfg(inner, {}, {'A': None}, {'N': 'N'})
    state.add_edge(nsdfg, 'A', state.add_write('A'), None, dace.Memlet('A[0:128]'))
    return sdfg, entry


sdfg, entry = outer_sdfg()
sdfg.validate()
print('pass result:', HoistLoopRangeCalls().apply_pass(sdfg, {}))
print('call still in nested increment:', contains_call(entry.map.range[0][2]))
A = np.zeros(128)
sdfg(A=A, N=np.int64(16))
print('ran; A[:16:4] =', A[:16:4])
