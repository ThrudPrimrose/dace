import dace
s = dace.SDFG.from_file('ib_canon.sdfg')
for sd in s.all_sdfgs_recursive():
    print(sd.name, {k: (type(v).__name__, str(v.dtype), v.shape, str(v.storage), v.transient) for k, v in sd.arrays.items()})
for n, g in s.all_nodes_recursive():
    if isinstance(n, dace.nodes.MapEntry): print('MAP', n.map.label, n.map.range, n.map.schedule)
    if isinstance(n, dace.nodes.Tasklet): print('T', n.label, n.code.as_string[:200], [(e.dst_conn, str(e.data)) for e in g.in_edges(n)], [(e.src_conn, str(e.data)) for e in g.out_edges(n)])
    if isinstance(n, dace.nodes.NestedSDFG): print('NS', n.label, n.symbol_mapping)
for sd in s.all_sdfgs_recursive():
    for cfr in sd.all_control_flow_regions():
        if type(cfr).__name__ == 'LoopRegion': print('LOOP', sd.name, cfr.loop_variable, cfr.init_statement.as_string, cfr.loop_condition.as_string, cfr.update_statement.as_string)
    for e in sd.all_interstate_edges():
        if e.data.assignments: print('ISE', sd.name, e.data.assignments)
