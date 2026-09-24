import dace
def dump_nested(sdfg, only_nested=True):
    for sd in sdfg.all_sdfgs_recursive():
        if only_nested and sd.parent_nsdfg_node is None: continue
        print('=== SDFG', sd.name, 'in', list(sd.parent_nsdfg_node.in_connectors) if sd.parent_nsdfg_node else '', 'out', list(sd.parent_nsdfg_node.out_connectors) if sd.parent_nsdfg_node else '')
        for st in sd.all_states():
            print(' -- state', st.label)
            for e in st.edges():
                def nm(n):
                    if isinstance(n, dace.nodes.AccessNode): return f'AN({n.data})#{st.node_id(n)}'
                    if isinstance(n, dace.nodes.Tasklet): return f'T[{n.code.as_string.strip()}]#{st.node_id(n)}'
                    return f'{type(n).__name__}#{st.node_id(n)}'
                print('   ', nm(e.src), e.src_conn, '->', nm(e.dst), e.dst_conn, ':', e.data)
