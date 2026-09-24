# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Move a loop that is a map's whole body outside the map: the inverse of ``MoveLoopIntoMap``."""

from typing import Optional

from dace import data as dt, sdfg as sd, symbolic
from dace.sdfg import nodes, utils as sdutil
from dace.sdfg import state as cf
from dace.transformation import transformation
from dace.transformation.passes.analysis import analysis


def reads_unwritten(state: sd.SDFGState, name: str, desc: dt.Data) -> bool:
    """Whether ``state`` may read an element of ``name`` that no write into the same access node covers."""
    for node in [n for n in state.data_nodes() if n.data == name]:
        writes = [e for e in state.in_edges(node) if not e.data.dynamic and e.data.wcr is None]
        reads = [e.data.get_src_subset(e, state) for e in state.out_edges(node) if not e.data.is_empty()]
        reads += [e.data.get_dst_subset(e, state) for e in state.in_edges(node) if e.data.wcr is not None]
        for subset in reads:
            if not any(
                    analysis.writes_whole_array(state, w, desc) or
                (isinstance(w.src, (nodes.Tasklet, nodes.AccessNode)) and w.data.get_dst_subset(w, state) == subset)
                    for w in writes):
                return True
    return False


def carried_across_iterations(loop: cf.LoopRegion, body: sd.SDFG) -> Optional[str]:
    """What keeps a value from one iteration of ``loop`` to the next inside ``body``, if anything does.

    Once the loop is outside the map, every iteration calls ``body`` afresh, so its transients and symbols start over.
    """
    assigned = {name for edge in loop.all_interstate_edges() for name in edge.data.assignments}
    inner = [r for r in loop.all_control_flow_regions() if r is not loop]
    assigned |= {r.loop_variable for r in inner if isinstance(r, cf.LoopRegion)}
    if any(edge.data.assignments for r in inner for edge in r.edges()):
        return 'a branch or an inner loop assigns a symbol'
    header = [c for c in (loop.init_statement, loop.loop_condition, loop.update_statement) if c is not None]
    reassigned = sorted(assigned & {str(s) for code in header for s in code.get_free_symbols()})
    if reassigned:
        return f'the loop bound {reassigned[0]} is reassigned inside the loop'
    # only a straight line of blocks assigns a symbol on every path to the blocks after the assignment
    straight = all(loop.in_degree(b) <= 1 and loop.out_degree(b) <= 1 for b in loop.nodes())
    defined: set[str] = set()
    for block in sdutil.dfs_topological_sort(loop):
        edges = loop.out_edges(block)
        reads = {str(s) for s in block.used_symbols(all_symbols=True)}
        reads |= {str(s) for e in edges for s in e.data.read_symbols()}
        carried = sorted((reads - defined) & assigned)
        if carried:
            return f'symbol {carried[0]} is carried across iterations'
        if straight:
            defined |= {name for edge in edges for name in edge.data.assignments}
    edge_reads = {str(s) for edge in loop.all_interstate_edges() for s in edge.data.read_symbols()}
    for name, desc in body.arrays.items():
        states = [s for s in loop.all_states() if any(n.data == name for n in s.data_nodes())]
        if desc.transient and states and (len(states) > 1 or name in edge_reads
                                          or reads_unwritten(states[0], name, desc)):
            return f'transient {name} is carried across iterations'
    return None


@transformation.explicit_cf_compatible
class MapLoopInterchange(transformation.SingleStateTransformation):
    """``map i: for t: B`` -> ``for t: map i: B``, for a map whose body is a nested SDFG holding only the loop.

    Map iterations are independent, so running every map iteration per loop iteration keeps the order each one
    sees. The move is legal when the loop's trip count is the same for every map iteration, nothing leaves an
    iteration early, and nothing inside the body carries a value between loop iterations. The map's state must
    hold nothing but the map, since the loop repeats the whole state.
    """

    map_entry = transformation.PatternNode(nodes.MapEntry)
    nested_sdfg = transformation.PatternNode(nodes.NestedSDFG)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.map_entry, cls.nested_sdfg)]

    def refusal(self, state: sd.SDFGState) -> Optional[str]:
        """Why the loop cannot move outside the map, or ``None``."""
        entry, body = self.map_entry, self.nested_sdfg
        blocks = body.sdfg.nodes()
        if len(blocks) != 1 or not isinstance(blocks[0], cf.LoopRegion):
            return 'the body is not exactly one loop'
        loop = blocks[0]
        exit_node = state.exit_node(entry)
        if any(e.dst is not body for e in state.out_edges(entry)) or any(e.src is not body
                                                                         for e in state.in_edges(exit_node)):
            return 'the map holds more than the nested SDFG'
        if any(not isinstance(n, nodes.AccessNode) for n in state.scope_children()[None]
               if n is not entry and n is not exit_node):
            return 'the state holds more than the map; the loop would repeat it'
        if any({e.src, e.dst}.isdisjoint({entry, exit_node}) for e in state.edges() if body not in (e.src, e.dst)):
            return 'the state holds a copy outside the map; the loop would repeat it'
        var = loop.loop_variable
        outer = state.sdfg
        taken = set(outer.symbols) | set(outer.arrays) | outer.free_symbols | outer.used_symbols(all_symbols=True)
        taken |= {name for edge in outer.all_interstate_edges() for name in edge.data.assignments}
        taken |= {r.loop_variable for r in outer.all_control_flow_regions() if isinstance(r, cf.LoopRegion)}
        if var in entry.map.params or var in taken:
            return f'the loop variable {var} is already defined outside the map'
        statements = [c for c in (loop.init_statement, loop.loop_condition, loop.update_statement) if c is not None]
        for name in sorted({str(s) for code in statements for s in code.get_free_symbols()} - {var}):
            if (name in entry.map.params or name in entry.in_connectors or name in body.sdfg.arrays
                    or str(body.symbol_mapping.get(name)) != name):
                return f'the loop bound {name} varies across map iterations'
        if any(
                isinstance(b, (cf.BreakBlock, cf.ContinueBlock, cf.ReturnBlock))
                for b in loop.all_control_flow_blocks()):
            return 'the loop leaves an iteration early'
        return carried_across_iterations(loop, body.sdfg)

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return self.refusal(graph) is None

    def apply(self, graph: sd.SDFGState, sdfg: sd.SDFG):
        state, body = graph, self.nested_sdfg
        loop = body.sdfg.nodes()[0]
        var = loop.loop_variable
        dtype = loop.new_symbols({})[var]
        # splice the loop body into the nested SDFG, which then runs one iteration per call
        blocks, edges, start = list(loop.nodes()), list(loop.edges()), loop.start_block
        body.sdfg.remove_node(loop)
        for block in blocks:
            loop.remove_node(block)
            body.sdfg.add_node(block, is_start_block=block is start)
        for edge in edges:
            body.sdfg.add_edge(edge.src, edge.dst, edge.data)
        # the loop variable is a free symbol of the body now, bound by the outer loop
        if var not in body.sdfg.symbols:
            body.sdfg.add_symbol(var, dtype)
        body.symbol_mapping[var] = symbolic.symbol(var, dtype)
        # wrap the map's state in the loop
        parent = state.parent_graph
        ins, outs = parent.in_edges(state), parent.out_edges(state)
        parent.add_node(loop, is_start_block=parent.start_block is state, ensure_unique_name=True)
        for edge in ins:
            parent.add_edge(edge.src, loop, edge.data)
        for edge in outs:
            parent.add_edge(loop, edge.dst, edge.data)
        parent.remove_node(state)
        loop.add_node(state, is_start_block=True)
        root = sdfg.root_sdfg
        sdutil.set_nested_sdfg_parent_references(root)
        root.reset_cfg_list()
