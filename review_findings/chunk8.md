# chunk8 findings: split_map_for_tile_remainder, stage_global_array_through_scalars, stride_map_by_tile_widths, tasklet_preprocessing_passes, utils/*

All reproducers are in `/tmp/claude-0/-home-user/1191ce82-d309-5728-89cd-bb12c458f12b/scratchpad/review/chunk8/`.
Run them from that directory with `source /home/user/.venv/bin/activate; PYTHONHASHSEED=0 python <file>`.
`drv.py` (helper) runs `VectorizeCPUMultiDim(VectorizeConfig(widths=(4,), target_isa='SCALAR', ...)).apply_pass`, the same way
`tests/passes/vectorization/test_vectorize_numerical_correctness.py` drives the pipeline. The numbers are compared
against numpy and against the same SDFG run without vectorization.

---

## F1. StageGlobalArrayThroughScalars drops a value when the next write to the same element is conditional (masked / IT)
- **Severity:** miscompile
- **Location:** `dace/transformation/passes/vectorization/stage_global_array_through_scalars.py:171-200` (`_downstream_same_subset_write`), used at `:391` and `:462-470`.
- **Root cause:** A write key counts as "superseded" when any later Tasklet-produced access node of the same array
  writes the same subset key. The pass then skips publishing that value to the bridge or out-connector. The check
  ignores whether the later write is unconditional. A masked write (`B(dyn)[i]`, the frontend's bare-if tasklet that
  `NormalizeMaskedWriteTasklets` rewrote to `__out = IT(cond, v)`) leaves the element unchanged on inactive lanes, so
  those lanes keep the stale pre-kernel value instead of the earlier write.
- **Reproducer:** `repro_stage_it.py`. The map body is `B[i] = A[i]+1; C[i] = B[i]*2; if M[i]: B[i] = 7`. The SDFG is hand-built in the frontend's masked-write shape.
  Observed output:
  ```
  unvectorized: B ok True C ok True
  tile ops 9
  vectorized:   B ok False C ok True
  B    [ 7. -1. -1. -1.  7.  7.  7.  7.  7.  7.  7. -1.  7. -1.  7. -1.]
  Bref [7. 1.2698 1.0410 1.0165 7. 7. 7. 7. 7. 7. 7. 1.0027 7. 1.0336 7. 1.1757]
  ```
  (-1 is B's input value.) Stage before/after dump: the `t0 -> B[i]` write is rerouted to `stage_B_rmw0` and nothing ever publishes it.
- **Proposed fix:** In `_downstream_same_subset_write`, only count an in-edge as an overwrite when it is unconditional:
  `not e.data.dynamic`, and the producing tasklet is not an `IT(...)` / bare-`if` conditional write. Otherwise keep walking.
  With that change the key is not superseded, so it gets published and the later IT write orders after it through the existing edges.

## F2. `recognize_map_reduction` accepts an element-wise in-place update as a loop-carried scalar reduction
- **Severity:** miscompile
- **Location:** `dace/transformation/passes/vectorization/utils/reductions.py:361-366` (`_scalar_slot`, `reads`, `writes`). Consumed by `LiftMapReductionToReduce._lift` (`rmw_only=True`), which `VectorizeMultiDim.apply_pass` calls.
- **Root cause:** An "accumulator" only has to be read at the map entry and written at the map exit with a
  single-element memlet of the same data name. The recognizer never checks that the read and write subsets are equal
  and independent of the map parameter. So `y[j] = y[j] + x[cols[j]]` (body NestedSDFG, frontend output) is classified
  as `op='+', accumulator='y'`. If the pre-map `y` node is seeded by a `0.0` tasklet (`y[0] = 0.0`), the lift replaces
  the per-element update with `y[0] = sum(x[cols])`.
- **Reproducer:** `repro_mapred.py` (plain `@dace.program`). Observed output:
  ```
  recognize_map_reduction -> ('+', 'y', 'y[j]', 'y[j]')
  unvectorized ok True
  Reduce nodes 1
  vectorized   ok False
  got      [136. 100. 100. ... 100.]
  expected [ 16. 115. 114. ... 101.]
  ```
- **Proposed fix:** In `recognize_map_reduction`, require `read_edge.data.subset == write_edge.data.subset` and
  `not (set(map_entry.map.params) & {str(s) for s in write_edge.data.subset.free_symbols})` before returning.
  Also, in `_lift`, check that each init edge's subset covers the accumulator subset.

## F3. SplitMapForTileRemainder(assume_even) hoists its extent guard out of the scope that defines the extent's symbols
- **Severity:** crash (generated C++ does not compile). It is also semantically wrong: the guard checks once instead of once per outer iteration.
- **Location:** `dace/transformation/passes/vectorization/split_map_for_tile_remainder.py:322` records the check with only
  the owning SDFG. `:463` emits it with `owner.add_state_before(owner.start_block, ...)`.
- **Root cause:** Each extent guard is placed in a new start state of the SDFG that owns the map. The extent can
  reference a symbol that exists only in a narrower scope: an enclosing map param (here the loop that
  `LoopToMap` turned into `single_state_body_map[k]`), a loop variable, or an interstate-assigned symbol. At the SDFG
  start that symbol is undeclared.
- **Reproducer:** `repro_guard_loopvar.py`: `for k in range(1, N): for i in dace.map[0:k]: B[k, i] = A[k, i] + 1`,
  `VectorizeConfig(assume_even=True)`. Observed output:
  ```
  tri tile_even_range_check ['if ((long long)(k) % 4 != 0 || (long long)(k) < 4) {']
  single_state_body_map 1:N
  tri_8__tile_main 0:k:4
  CompilationError: ... tri.cpp:33:29: error: 'k' was not declared in this scope
  ```
- **Proposed fix:** Record `(state, map_entry)` along with each check. Emit only extents whose free symbols are all in
  `owner.free_symbols` (the SDFG arguments) at the start block. For any other extent, either emit the guard in a
  tasklet just before the map inside its enclosing scope, or skip the guard.

## F4. StageGlobalArrayThroughScalars (flat map-body path) creates a cyclic state when a read-only subset has no outer source
- **Severity:** crash (InvalidSDFGError). It is raised from `VectorizeCPUMultiDim` even though the map is not a tile candidate.
- **Location:** `dace/transformation/passes/vectorization/stage_global_array_through_scalars.py:223-245` (`_find_outer_source`
  falls back to `return outer_drain`), used at `:421-427`.
- **Root cause:** The map-body bridge `B` is written at `B[i,0]` and read at `B[i,1]`, and no outer `B` node feeds the
  map. `_find_outer_source` then returns the map's own output node `outer_drain`, and `_add_scoped_path` wires
  `outer_drain -> MapEntry -> stage_B_r1`. The result is the cycle `B -> MapEntry -> ... -> MapExit -> B`. The pass runs on
  every map, including maps the vectorizer leaves scalar (here the body has a C++ tasklet, so it is never nested).
  So a valid, non-vectorizable map crashes the whole pipeline.
- **Reproducer:** `repro_stage_cycle.py` (hand-built SDFG). Observed output:
  ```
  unvectorized runs; C ok True B[:,0] ok True
  StageGlobalArrayThroughScalars -> 1
  after pass: InvalidSDFGError: State should be acyclic but contains cycles (at state block)
  VectorizeCPUMultiDim raised InvalidSDFGError: State should be acyclic but contains cycles (at state block)
  ```
- **Proposed fix:** When no outer source exists that is distinct from `outer_drain`, refuse the occurrence. Do this before
  any mutation, i.e. move the lookup and check to the top of `_apply_multi` and return False. Never return `outer_drain`
  as a source. A cheaper option: when `read_only_keys` is non-empty and the source would be `outer_drain`, return False.

## F5. PowerOperatorExpansion: uncapped unrolling of a literal integer exponent crashes the pipeline
- **Severity:** crash (plus a perf/accuracy trap for moderate exponents: `x**100` becomes 99 dependent multiplies instead of one `pow`)
- **Location:** `dace/transformation/passes/vectorization/tasklet_preprocessing_passes.py:91-96` (`PowerOperatorExpander._expand_pow`).
- **Root cause:** Any integer literal `n > 1` is unrolled into a left-nested chain of `n-1` `BinOp(Mult)`. `ast.unparse`
  recurses once per level, so `A[i] ** 5000` raises RecursionError inside `VectorizeCPUMultiDim`. It is not caught as
  a refusal.
- **Reproducer:** `repro_bigpow.py`: `B[i] = A[i] ** 5000`. Observed output: `vectorizer raised RecursionError maximum recursion depth exceeded`.
- **Proposed fix:** Only expand small exponents, e.g. `2 <= n <= 4` (or 8), and leave larger ones as `**` for the
  ipow/pow lowering. Alternatively, expand by repeated squaring to O(log n) depth.

---

## Unverified suspicions
- `pass_invariants.tile_main_map_step_is_widths` / `no_strided_map_param_in_surviving_condition` iterate `sd.states()` (top-level only), so maps inside LoopRegion/ConditionalBlock states are never checked. This is a toothless post-condition, not an observed miscompile.
- `StageGlobalArrayThroughScalars` groups subsets by string key. Two spellings that alias at runtime (`B[i]` vs `B[j]` with `j == i`) would be treated as independent keys, so a read-only key is sourced from the pre-map value instead of this iteration's write.
- `TileNameScheme.is_tile_transient` matches any name ending in `_tile`/`_tile_idx`, which includes user (non-transient) arrays.

Checked and found correct: K=1/2/3 remainder splits (masked_tail, scalar_postamble, tile_k1) on symbolic, offset and empty ranges;
strided source maps (normalized upstream); reduction-param narrowing (`y[i] += A[i, j]` at widths (4,4)); RMW chains,
WAW with an ordering edge, multi-subset pure writers and cross-state bridges in Stage; `write_subset_is_injective` on
Mod/floor/Min/Abs/nonlinear indices (always conservative); `x ** 2.0` on int32 input (casts keep it exact).
