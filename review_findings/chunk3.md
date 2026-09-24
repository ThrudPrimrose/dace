# chunk3 findings (canonicalize: normalize_* / partition / perfect nesting / pipeline / privatize / prune)

Reproducers are in `/tmp/claude-0/-home-user/1191ce82-d309-5728-89cd-bb12c458f12b/scratchpad/review/chunk3/`.
Run each with `source /home/user/.venv/bin/activate; PYTHONHASHSEED=0 python <script>` (the GCC 13 warning lines are left out of the output below).

Where the passes run in the pipeline: `NormalizeNegativeStride` runs in the `clean` and `lower` stages. `PerfectLoopNesting` runs in `fission` (on by default). `NormalizeMapBody` runs in `fuse`. `PruneUnreferencedTransients` runs in cleanup and `end`. `PruneAndInlineNestedSDFGs` runs in structural cleanup, and `NormalizeFloorDivision` in `end`. `NormalizeLoopAndMapOrigin` is off by default (`CPU_DEFAULTS`/`GPU_DEFAULTS`). `NormalizeStridedMaps` runs at vectorizer entry (`vectorize_multi_dim.py:1340`), and `NormalizeLoopsAndMaps._normalize_map` is used by `iteration_domain.align_maps_to_unit_step` (FuseMaps). `PrivatizeReductionAccumulator` and `PartitionGuardedLoop` are not wired into the pipeline.

---

## 1. NormalizeNegativeStride gives an empty loop one iteration (trip count uses truncating `int_floor`)
- **Severity:** miscompile (default pipeline)
- **Location:** `dace/transformation/passes/canonicalize/normalize_negative_stride.py:113` (and the condition it feeds, line 149)
- **Root cause:** The trip count is `int_floor(start - end_incl, k) + 1`, but `int_floor` emits C `/`, which truncates toward zero (see `symbolic.py:1721`). When the loop runs zero times, `start - end_incl` is negative. For `-k < start - end_incl < 0` the truncated quotient is 0 instead of -1, so the trip count becomes 1 and the body runs once. `range(N, M, -2)` with `N == M` is the plain case. Stride -1 is unaffected; any `|stride| >= 2` is affected.
- **Reproducer:** `negstride_trip.py` (direct pass), plus `pipeline_checks.py neg` (full `canonicalize`)
  ```
  before: i i = N (i > M) i = (i + (- 2))
  pass returned 1
  after: _loop_pos_0 _loop_pos_0 = 0 (_loop_pos_0 < (int_floor((((- M) + N) - 1), 2) + 1)) _loop_pos_0 = (_loop_pos_0 + 1)
  N=5,M=5 dace: [5] numpy: []
  N=5,M=8 dace: [] numpy: []
  ```
  full pipeline: `N=5,M=5 written: [5] numpy: []`
- **Proposed fix:** Avoid the division altogether. Keep the original comparison with the rebound value substituted, e.g. `loop_condition = f"({start}) + ({stride}) * {new_var} > ({end_excl})"` (the same op the original condition used). Alternatively, use the exact ceiling form `Max(0, int_ceil(start - end_incl + 1, k))`.

## 2. NormalizeStridedMaps / NormalizeLoopsAndMaps give an empty strided map or loop one iteration (same truncation)
- **Severity:** miscompile (vectorizer entry `vectorize_multi_dim.py:1340`, FuseMaps `align_maps_to_unit_step`)
- **Location:** `dace/transformation/passes/canonicalize/normalize_loops_and_maps.py:105` (maps). Line 174 (loops) has the same root cause.
- **Root cause:** The map `b:e:s` becomes `0:int_floor(e - b, s):1` (inclusive end). For an empty map, `e - b` lies in `(-s, 0)`, and C truncation gives 0 instead of -1, so one iteration runs at `p = b`. The docstring's "symbols non-negative" argument does not help: `e - b` is negative for `dace.map[3:N:2]` at `N = 3`, even though N is non-negative.
- **Reproducer:** `strided_map_empty.py`
  ```
  before: 3:N:2
  pass returned 1
  after : 0:int_floor(N - 4, 2) + 1
  N=3 dace written: [3]  numpy written: []
  ```
- **Proposed fix:** Clamp the end so the map stays empty: `(0, int_ceil(e - b + 1, s) - 1, 1)` with a `Max(-1, ...)` guard, or use `int_floor` only when `e - b >= 0` can be proven and refuse otherwise. Apply the same clamp to `n` in `_normalize_loop`.

## 3. NormalizeLoopAndMapOrigin (and every user of `repl_tasklets_on_node_list`) corrupts tasklet code
- **Severity:** crash, or silent miscompile
- **Location:** `dace/transformation/passes/offset_loop_and_maps.py:111-113` (Python path) and `:90` (token path). They are reached from `normalize_loop_and_map_origin.py:215` / `:254` and `normalize_loops_and_maps.py:128` / `:178`.
- **Root cause:** Every Python tasklet in scope is rewritten as `code.split(" = ")[0] + " = " + pycode(SymExpr(code.split(" = ")[-1]))`. This also happens to tasklets that do not mention the parameter. For a multi-statement tasklet this keeps the first LHS and the last RHS and throws away everything in between. The fallback `token_replace_dict` (used for C++, and whenever SymExpr fails) splits only on whitespace and brackets. As a result `i;`, `i+1` and `a[i+1]` are never matched, and the parameter reference is not shifted while the memlets are.
- **Reproducers:**
  - `origin_multistmt.py`: tasklet `t = (a * 2.0)\nb = (t + 1.0)` becomes `t = (t + 1.0)`:
    ```
    before: ['t = (a * 2.0)\nb = (t + 1.0)']
    pass returned 1
    after: ['t = (t + 1.0)']
    ... prog.cpp:15:19: error: use of 't' before deduction of 'auto'
    ```
  - `origin_cpp_tasklet.py` (C++ tasklet `b = i;` over `i = 1:10`, writing `B[i]`):
    ```
    pass returned 1
    map range: 0:9  write memlet: B[i + 1]  tasklet: 'b = i;'
    ```
    This computes `B[k] = k - 1` instead of `B[k] = k`.
- **Proposed fix:** Rewrite tasklets with an AST transformer on the Python code (the `ASTFindReplace` used by `replace_dict`), and only for tasklets whose free symbols include a key. For C++, refuse the rebase, or use a word-boundary regex (`\bi\b`) that skips connector names. Never splice on `" = "`.

## 4. NormalizeMapBody binds a consumer's carrier array to the producer's without comparing subsets
- **Severity:** miscompile (`fuse` stage)
- **Location:** `dace/transformation/passes/canonicalize/normalize_map_body.py:110-122` (legality) and `:282-286` (rename)
- **Root cause:** `shared_carrier_connectors` accepts a carrier `keep -> X -> drop` without checking that the subset `keep` writes and the subset `drop` reads are the same region. `_merge_siblings` then renames drop's inner array onto keep's (`replace_datadesc_names(tail, {renamed: keep_conn})`). After that, drop's index 0 addresses the start of keep's written window, not the element drop was reading. The shapes can differ too, and the base keeps keep's descriptor.
- **Reproducer:** `mapbody_carrier_subset.py`: keep writes `tmp[0:2]` and drop reads `tmp[1]`.
  ```
  before pass: [10.0, 30.0, 50.0, 70.0] expected [10.0, 30.0, 50.0, 70.0]
  pass returned 1
  after pass : [0.0, 20.0, 40.0, 60.0] expected [10.0, 30.0, 50.0, 70.0]
  ```
- **Proposed fix:** In `shared_carrier_connectors`, refuse (`return None`) unless drop's in-edge subset equals keep's out-edge subset for that carrier, and the two inner descriptors have equal shape. An alternative is to offset the tail's accesses, but refusing is the minimal fix.

## 5. NormalizeMapBody drops the dropped sibling's symbol binding when an inner name collides
- **Severity:** miscompile (`fuse` stage)
- **Location:** `dace/transformation/passes/canonicalize/normalize_map_body.py:345-346` (with the assumption stated at `:61-66` and `:291-293`)
- **Root cause:** Symbols are never uniquified. `keep.symbol_mapping.setdefault(k, v)` keeps keep's binding whenever both siblings use the same inner symbol name, even when drop binds it to a different outer value (`{'k': 'i'}` versus `{'k': 'i + 1'}`). The merged tail body then reads keep's value.
- **Reproducer:** `mapbody_symbol_mapping.py`
  ```
  before: C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  pass returned 1
  after: C = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  expected [1.0 .. 8.0]
  ```
- **Proposed fix:** Before splicing, compare each overlapping key. If `str(keep.symbol_mapping[k]) != str(drop.symbol_mapping[k])`, either rename `k` in `tail` (`tail.replace_dict({k: fresh})`, then map `fresh -> v`) or refuse the merge.

## 6. NormalizeMapBody merges some siblings, then refuses and reports "no change"
- **Severity:** a refusing pass that still mutates. It also desyncs the pipeline's `dirty` flag (`pipeline.py:2426-2435`), so the next structural cleanup is skipped.
- **Location:** `dace/transformation/passes/canonicalize/normalize_map_body.py:264-267`, and `:254` (count only on True)
- **Root cause:** `_merge_siblings` folds siblings into `keep` one by one. When a later sibling is refused, it returns `False` after the earlier merges have already been applied. It also skips `_dedup_boundary_aliases` and the `blk.sdfg = base` fix-up. `apply_pass` counts only a `True` result, so it returns `None` for a changed SDFG.
- **Reproducer:** `mapbody_partial_refusal.py` (siblings n0: X->Y, n1: A->C, n2: A->X, where n2 writes what n0 reads)
  ```
  nested SDFGs before: 3
  pass returned: None
  nested SDFGs after: 2
  SDFG unchanged: False
  valid
  ```
- **Proposed fix:** Run the `shared_carrier_connectors` check for every drop before mutating anything. A refused drop should be removed from the list, not abort the merge. After at least one merge, always run the tail of `_merge_siblings` (dedup and re-homing) and return True.

## 7. Pipeline turns off UniqueLoopIterators' post-value epilogue, so a read of the iterator after the loop becomes a free symbol
- **Severity:** crash (the SDFG gains a spurious required argument; call fails). Inside a nested SDFG it would read an unbound or stale value instead.
- **Location:** `dace/transformation/passes/canonicalize/pipeline.py:808-811` (rationale at `:800-807`)
- **Root cause:** Every `UniqueLoopIterators` in the recipe is built with `assign_loop_iterator_post_value=False`. The comment's justification ("canonicalize already rewrites every use site to the unique name") does not hold for a use after the loop. The rename is scoped to the loop's subtree, so `C[0] = i` after `for i in range(N)` still reads `i`, and nothing binds it any more. Bisecting the stage list without compiling, the first unit that makes `i` free is `clean / UniqueLoopIterators`. `PerfectLoopNesting` (`perfect_loop_nesting.py:295`) makes the same choice.
- **Reproducer:** `pipeline_checks.py postloop` (full `canonicalize(validate=True)`, which validates cleanly), `bisect_free_i.py`, and `pln_postloop.py` (PerfectLoopNesting alone)
  ```
  KeyError: 'Missing program argument "i"'           # pipeline_checks.py postloop (uncanonicalized: C = 5)
  initial free ['N']
  first introduces free i: clean UniqueLoopIterators # bisect_free_i.py
  free symbols: {'N', 'i'}                           # pln_postloop.py, after PerfectLoopNesting
  ```
- **Proposed fix:** Construct the pipeline's instances with `assign_loop_iterator_post_value=True`. The pass already emits the epilogue only when `_post_value_needed` finds a read after the loop, so no dead states are added otherwise. Do the same in `PerfectLoopNesting.apply_pass`.

## 8. PrivatizeReductionAccumulator: a reader in the same state sees the pre-reduction value
- **Severity:** miscompile (the pass is not wired into the pipeline; its TODO is at `pipeline.py:1631-1642`)
- **Location:** `dace/transformation/passes/canonicalize/privatize_reduction_accumulator.py:178-189`
- **Root cause:** In the cross-state pattern, the MapExit's edge into `arr_node` is removed and the writeback is moved to a new state after the current one. `arr_node` is kept whenever it still has out-edges. A downstream consumer in the same state that reads `arr_node` therefore reads `arr` before the reduction has been written back.
- **Reproducer:** `privatize_stale_read.py`
  ```
  before: s = 29.0, out = 58.0   expected s = 29.0, out = 58.0
  pass returned 1
  after: s = 29.0, out = 2.0   expected s = 29.0, out = 58.0
  ```
- **Proposed fix:** Refuse when `state.out_degree(arr_node) > 0` (or when any other node in the state reads `arr`). Alternatively, write back in the same state by routing `new_scalar_an -> arr_node`, as the in-state branch does. The in-state branch (lines 152-167) also leaves `seed_an` unordered against the map's WCR write and should chain it as the WCR target's predecessor.

---

## Unverified suspicions
- `NormalizeFloorDivision` only covers descriptors, map ranges and memlet subsets. A `sympy.floor` in a NestedSDFG `symbol_mapping` value, an interstate assignment, or a loop or branch condition still reaches codegen unnormalized. I found no pass that produces one there.
- `NormalizeNegativeStride` / `NormalizeLoopAndMapOrigin` change the iterator's value after the loop compared with the unoptimized SDFG. After the rewrite it is the last value; before, it was the C exit value (2 before and 3 after in `negstride_postloop.py`). The rewritten value happens to match Python, so there is no mismatch against numpy.
- `NormalizeMapBody` merges a sibling that reads from the MapEntry what `keep` writes to the MapExit, and sequences it write-then-read. Whether the original intra-iteration order was defined is unclear, so I did not report it.
