import dace
from dace.transformation.passes.vectorization import stage_global_array_through_scalars as sg, split_map_for_tile_remainder as sm, stride_map_by_tile_widths as st
SNAP = {}
def wrap(cls, key):
    orig = cls.apply_pass
    def ap(self, sdfg, res):
        import copy
        SNAP[key + '_before'] = copy.deepcopy(sdfg)
        r = orig(self, sdfg, res)
        SNAP[key + '_after'] = copy.deepcopy(sdfg)
        print(f'[{key}] ->', r)
        return r
    cls.apply_pass = ap
wrap(sg.StageGlobalArrayThroughScalars, 'stage')
wrap(sm.SplitMapForTileRemainder, 'split')
wrap(st.StrideMapByTileWidths, 'stride')
