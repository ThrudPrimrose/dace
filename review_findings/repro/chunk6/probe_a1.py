import sys
from dace.transformation.passes.vectorization import insert_tile_load_store as its
orig = its.InsertTileLoadStore._maybe_stage_tilestore_to_output
def wrapped(self, st, b, c, iv, orig_edge=None):
    r = orig(self, st, b, c, iv, orig_edge)
    print('A1 CALL', b.data, '->', c.data, 'orig', orig_edge.data if orig_edge is not None else None, '->', r,
          [str(e.data) for e in st.in_edges(c) if type(e.src).__name__ == 'TileStore'])
    return r
its.InsertTileLoadStore._maybe_stage_tilestore_to_output = wrapped
sys.argv = ['batch7.py'] + sys.argv[1:]
exec(open(sys.argv[1] if False else 'batch8.py').read())
