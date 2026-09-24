# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import copy

import numpy as np

import dace
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion
from dace.transformation.interstate import MapLoopInterchange, MoveLoopIntoMap

N, T, K = (dace.symbol(s) for s in 'NTK')


@dace.program
def map_loop(A: dace.float64[T, N]):
    for i in dace.map[0:N]:
        for t in range(1, T):
            A[t, i] = A[t - 1, i] + 1.0


@dace.program
def map_triangle_loop(A: dace.float64[N, N]):
    for i in dace.map[0:N]:
        for t in range(i + 1, N):
            A[t, i] = A[t - 1, i] + 1.0


@dace.program
def map_loop_break(A: dace.float64[T, N]):
    for i in dace.map[0:N]:
        for t in range(1, T):
            if A[t - 1, i] > K:
                break
            A[t, i] = A[t - 1, i] + 1.0


@dace.program
def loop_map(A: dace.float64[T, N]):
    for t in range(1, T):
        for i in dace.map[0:N]:
            A[t, i] = A[t - 1, i] + 1.0


@dace.program
def copy_then_map_loop(B: dace.float64[N], tmp: dace.float64[N]):
    tmp[:] = B
    for i in dace.map[0:N]:
        for t in range(T):
            B[i] = B[i] + tmp[i]


@dace.program
def map_loop_double_buffer(B: dace.float64[N]):
    for i in dace.map[0:N]:
        for t in range(3):
            tmp = np.ndarray([2], dtype=np.float64)
            tmp[t % 2] = t + 1
            B[i] = B[i] + tmp[(t + 1) % 2]


def mapped_loop(condition: str, code: str, assignments: dict, symbols: dict) -> dace.SDFG:
    """``map i: for (t = 0; condition; t++) { B[i] = code(x=B[i]); assignments }``."""
    body = dace.SDFG('body')
    body.add_array('B', [1], dace.float64)
    loop = LoopRegion('tloop', condition, 't', 't = 0', 't = t + 1')
    body.add_node(loop, is_start_block=True)
    st = loop.add_state('st0', is_start_block=True)
    loop.add_edge(st, loop.add_state('st1'), dace.InterstateEdge(assignments=assignments))
    tasklet = st.add_tasklet('acc', {'x': None}, {'y': None}, code)
    st.add_edge(st.add_access('B'), None, tasklet, 'x', dace.Memlet('B[0]'))
    st.add_edge(tasklet, 'y', st.add_access('B'), None, dace.Memlet('B[0]'))
    sdfg = dace.SDFG('mapped_loop')
    sdfg.add_array('B', ['N'], dace.float64)
    for name in symbols:
        body.add_symbol(name, dace.int64)
        sdfg.add_symbol(name, dace.int64)
    state = sdfg.add_state('main')
    me, mx = state.add_map('m', {'i': '0:N'})
    nsdfg = state.add_nested_sdfg(body, {'B': None}, {'B': None}, symbols)
    state.add_memlet_path(state.add_access('B'), me, nsdfg, dst_conn='B', memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(nsdfg, mx, state.add_access('B'), src_conn='B', memlet=dace.Memlet('B[i]'))
    return sdfg


def run(sdfg: dace.SDFG, **symbols) -> np.ndarray:
    A = np.random.default_rng(0).random((symbols['T'], symbols['N']))
    sdfg(A=A, **symbols)
    return A


def test_the_loop_moves_outside_the_map_and_computes_the_same_values():
    sdfg = map_loop.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)

    assert sdfg.apply_transformations(MapLoopInterchange) == 1

    (loop, ) = sdfg.nodes()
    (state, ) = loop.nodes()
    assert isinstance(loop, LoopRegion) and loop.loop_variable == 't'
    assert any(isinstance(n, nodes.MapEntry) for n in state.nodes())
    assert np.allclose(run(sdfg, N=5, T=4), run(reference, N=5, T=4), rtol=0, atol=0)


def test_the_interchange_undoes_move_loop_into_map():
    sdfg = loop_map.to_sdfg(simplify=True)
    reference = copy.deepcopy(sdfg)
    assert sdfg.apply_transformations(MoveLoopIntoMap) == 1

    assert sdfg.apply_transformations(MapLoopInterchange) == 1

    assert [type(b) for b in sdfg.nodes()] == [LoopRegion]
    assert np.allclose(run(sdfg, N=5, T=4), run(reference, N=5, T=4), rtol=0, atol=0)


def test_a_loop_whose_bound_reads_the_map_parameter_stays_inside():
    """Each map iteration runs a different trip count, which one loop outside the map cannot express."""
    sdfg = map_triangle_loop.to_sdfg(simplify=True)

    assert sdfg.apply_transformations(MapLoopInterchange) == 0


def test_a_loop_that_breaks_from_inside_a_branch_stays_inside():
    """A break ends one map iteration's loop early; outside the map it would end every iteration's."""
    sdfg = map_loop_break.to_sdfg(simplify=True)

    assert sdfg.apply_transformations(MapLoopInterchange) == 0


def test_a_copy_in_the_map_state_keeps_the_loop_inside():
    sdfg = copy_then_map_loop.to_sdfg(simplify=True)

    assert sdfg.apply_transformations(MapLoopInterchange) == 0

    B = np.ones(4)
    sdfg(B=B, tmp=np.zeros(4), N=4, T=3)
    assert np.array_equal(B, [4, 4, 4, 4])


def test_a_symbol_assigned_in_one_iteration_and_read_in_the_next_keeps_the_loop_inside():
    sdfg = mapped_loop('t < T', 'y = x + s', {'s': 't'}, {'s': 's', 'T': 'T'})

    assert sdfg.apply_transformations(MapLoopInterchange) == 0

    B = np.zeros(3)
    sdfg(B=B, N=3, T=4, s=100)
    assert np.array_equal(B, [103, 103, 103])


def test_a_loop_whose_bound_the_body_reassigns_stays_inside():
    sdfg = mapped_loop('t < M', 'y = x + M', {'M': '2'}, {'M': 'M'})

    assert sdfg.apply_transformations(MapLoopInterchange) == 0

    B = np.zeros(3)
    sdfg(B=B, N=3, M=5)
    assert np.array_equal(B, [5 + 2, 5 + 2, 5 + 2])


def test_a_loop_variable_the_outer_sdfg_assigns_on_an_edge_stays_inside():
    sdfg = mapped_loop('t < 3', 'y = x + 1', {}, {})
    sdfg.add_state_before(sdfg.start_block, is_start_block=True, assignments={'t': '7'})
    sdfg.add_array('C', [1], dace.float64)
    after = sdfg.add_state_after(sdfg.sink_nodes()[0])
    after.add_edge(after.add_tasklet('rd', {}, {'y': None}, 'y = t'), 'y', after.add_access('C'), None,
                   dace.Memlet('C[0]'))

    assert sdfg.apply_transformations(MapLoopInterchange) == 0

    C = np.zeros(1)
    sdfg(B=np.zeros(2), C=C, N=2)
    assert C[0] == 7


def test_a_transient_element_read_one_iteration_after_its_write_keeps_the_loop_inside():
    sdfg = map_loop_double_buffer.to_sdfg(simplify=True)

    assert sdfg.apply_transformations(MapLoopInterchange) == 0
