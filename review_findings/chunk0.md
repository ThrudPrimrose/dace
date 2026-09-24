# chunk0 findings (canonicalize: annotate_loop_kinds, arg_max_lift, assume_symbols_nonnegative, cascade_iedge_assignments_up, collapse_noop_cast, dead_carried_store, debug, distribute_producer_consumer, eliminate_trivial_tasklets, empty_state_elimination)

Reproducers live in `/tmp/claude-0/-home-user/1191ce82-d309-5728-89cd-bb12c458f12b/scratchpad/review/chunk0/`.
Run each one with `source /home/user/.venv/bin/activate; PYTHONHASHSEED=0 python <file>` from that directory. In every
end-to-end case I also re-ran the full `canonicalize()` with only the named pass stubbed out, and the result was
correct. That rules out other passes as the cause.

---

## 1. CascadeInterstateEdgeAssignmentsUp hoists a loop-body symbol assignment past earlier reads of that symbol

- **Severity:** miscompile (reached from the Python frontend through the full `canonicalize()`)
- **Files and lines:** `dace/transformation/passes/canonicalize/cascade_iedge_assignments_up.py`
  - `_block_reads_symbols` at 235-266 and `_block_writes` at 269-286
  - the in-edge skip in `_legal_to_hoist_into` at 356-358
  - `_place_assignment_at` at 497-498
- **Root cause.** Three separate gaps in the legality check:
  1. `ConditionalBlock` is not a subclass of `ControlFlowRegion`. Both helpers therefore return empty read and write
     sets for a `ConditionalBlock`: its branch conditions and its branch bodies are never examined. The L3 check (a
     predecessor reads `key`) and the L2/L4 checks (a predecessor writes `key` or a symbol of `rhs`) all pass silently.
  2. L3 only looks at the contents of blocks. It never looks at interstate-edge right-hand sides or conditions that
     read `key`, whether in the loop body before the origin or in the parent region.
  3. The parent-edge scan skips the loop's own in-edges (`if e.dst is child: continue`). `_place_assignment_at` then
     writes `e.data.assignments[key] = rhs` on those edges and silently overwrites a different pre-loop value such as
     `m = 0`.

  The result: a flag that is reset after its first use inside the loop gets its new value on iteration 0 as well.
- **Reproducers and observed output:**
  - `cascade_frontend.py` (frontend: `m = 0; for i: if m > 5: A[i]=1 else: A[i]=2; m = K + 1`, then full canonicalize)
    ```
    numpy : [2 1 1 1 1 1]
    dace  : [1 1 1 1 1 1]
    MISMATCH
    ```
    With the pass stubbed out (`cascade_frontend_noop.py`): `dace (cascade disabled): [2 1 1 1 1 1]`.
  - `cascade_frontend_trace.py` shows the first move, at stage 51 (`cascade_iedges_up`):
    ```
    before: SDFGState block -> LoopRegion for_13 {'m': '0'}
            ConditionalBlock if_14 -> SDFGState assign_18_8 {'m': 'K + 1'}
    after:  SDFGState block -> LoopRegion for_13 {'m': 'K + 1'}
    ```
    Gaps (1) and (3) together cause this case.
  - `cascade_conditional_read.py` (hand-built; covers gap 1 only, `m` is an argument symbol):
    ```
    before pass: [2 1 1 1]
    after  pass: [1 1 1 1]
    MISMATCH
    ```
  - `cascade_overwrite_inedge.py` (hand-built; covers gaps 2 and 3, the read `j = m + 1` is on an edge):
    ```
    edge init -> L_iedge_hoist {'m': 'K + 1'} ; before pass: [ 1 12 12 12]  after pass: [12 12 12 12]
    MISMATCH
    ```
- **Proposed minimal fix:**
  - In `_block_reads_symbols` and `_block_writes`, handle `AbstractControlFlowRegion` (or `ConditionalBlock`
    explicitly): add its branch conditions' free symbols and recurse into its branches. An unknown block type should
    fail closed (`None`).
  - For L3, include `names_read_by` of every assignment right-hand side and every condition on interstate edges:
    - edges among `origin`'s predecessors in `edge_region`;
    - edges among `child`'s predecessors in `parent`, including `child`'s own in-edges;
    - the loop header of each crossed `LoopRegion` (`init_statement`, `loop_condition`, `update_statement`).
  - In `_legal_to_hoist_into`, refuse when an in-edge of `child` already assigns `key` with a different right-hand
    side. Do not skip those edges.

---

## 2. DeadCarriedStoreElimination drops a store that is never overwritten: indices on the other axes are never compared

- **Severity:** miscompile (frontend, full `canonicalize()`)
- **Files and lines:** `dace/transformation/passes/canonicalize/dead_carried_store.py`
  - `constant_offset_on_axis` at 66-88
  - the pairing loop in `find_killed_store` at 174-179
- **Root cause.** `constant_offset_on_axis` returns only `(axis, offset)` for the axis that carries the loop variable.
  It accepts any loop-invariant index on the other axes and then throws those indices away. `find_killed_store`
  pairs a "dead" store with a "kill" store using only the carried axis and the offset. So `a[i+1, 1]` is treated as
  killed by `a[i, 0]`, even though the two stores write different columns. The module docstring promises "matching
  non-scan indices", but nothing checks them.
- **Reproducer:** `dead_carried_other_axis.py`, the loop `for i in range(N-1): a[i,0] = b[i]; a[i+1,1] = c[i]`.
  - Standalone (`to_sdfg(simplify=True)`, then the pass):
    ```
    pass returned 1
    max |diff| column 1: 0.8107891472882043
    MISMATCH
    ```
  - `python dead_carried_other_axis.py canonicalize` (full pipeline): `max |diff| column 1: 0.81...  MISMATCH`.
  - With the pass stubbed out (`dead_carried_noop_check.py`): `with pass disabled: ok`.
- **Proposed minimal fix:** have `constant_offset_on_axis` also return the tuple of non-carried index expressions.
  In `find_killed_store`, require dead and kill to match exactly on those expressions
  (`symbolic.simplify(x - y) == 0` for each one). Do the same when classifying reads in `reads_are_clear`: a read on
  a different, provably distinct non-carried index can be skipped, and a read that cannot be proven distinct must
  refuse.

---

## 3. DeadCarriedStoreElimination peels tail iterations without a trip-count guard, so a short or zero-trip loop runs out of bounds

- **Severity:** miscompile, including out-of-bounds reads and writes (frontend, full `canonicalize()`)
- **Files and lines:** `dace/transformation/passes/canonicalize/dead_carried_store.py:357-364`, the
  `LoopPeeling().apply_to(..., verify=False, options={'count': distance, 'begin': False})` call.
- **Root cause.** The pass peels `distance` iterations off the tail and relies on the loop running at least
  `distance` times. The `max_peel` docstring even says so: "bounds the assumption that the loop runs more times than
  it peels". But `LoopPeeling` emits the peeled iterations unconditionally, and neither a guard nor a
  `record_assumption` trap is added. When the loop runs fewer times (for example `range(N - 2)` with `N = 2`, which is
  zero trips), the peeled iteration still executes, with `i = -1`.
- **Reproducers:**
  - `dead_carried_short_trip.py` (standalone pass; the program is `for i in range(N-2): a[i+1] = b[i]*2; a[i+2] = c[i]`):
    ```
    pass returned 1
    2 MISMATCH dace=[0.    0.    0.041 0.017 0.813 0.913] ref=[0.637 0.27  0.041 0.017 0.813 0.913]
    3 ok
    4 ok
    ```
  - `dead_carried_short_trip_canon.py` (full canonicalize): the same `N=2 MISMATCH`.
  - `dead_carried_short_trip_canon_noop.py` (the pass stubbed out): `2 ok / 3 ok / 5 ok`.
- **Proposed minimal fix:** before peeling, refuse unless the trip count is provably `>= distance`. Alternatively,
  wrap the peeled tail and the shortened loop in a `ConditionalBlock` on `trip >= distance`, with the original loop
  as the else branch. Recording an assumption for `AssumeSymbolConstraints` is not enough: it would abort programs
  that are valid.

---

## 4. DistributeProducerConsumerLoop reorders a producer after its consumer when a merge class is not contiguous

- **Severity:** miscompile (frontend, full `canonicalize()`, stage 52 `distribute`)
- **Files and lines:** `dace/transformation/passes/canonicalize/distribute_producer_consumer.py:164-168`, the group
  ordering; `LoopFission._fission_blocks` then emits the groups in that order.
- **Root cause.** Union-find can merge blocks that are not adjacent: B0 and B2 merge because B2 writes `a`, which B0
  reads. The block between them, B1, is left alone because B1 → B2 is an aligned forward producer. The groups are
  then sorted by their first member, which emits `for i:{B0;B2}` before `for i:{B1}`. B2 therefore reads `u` before B1
  has produced it.

  Allen-Kennedy distribution requires emitting the condensed dependence graph in topological order, or,
  equivalently, closing each class over the interval between its first and last member. Sorting by first position
  does neither.
- **Reproducers:**
  - `distribute_noncontiguous.py` (hand-built, standalone pass):
    ```
    pass returned 1
       L_fis0 ['B0', 'B2']
       L_fis1 ['B1']
    a before pass: [2.826 2.213 2.459 2.087 2.87 ]
    a after  pass: [1. 1. 1. 1. 1.]
    MISMATCH
    ```
  - `distribute_frontend.py` (frontend: `t[i]=a[i]; if i%2==0: u[i]=b[i]*2 else: u[i]=b[i]*3; a[i]=u[i]+1`, full
    canonicalize): `MISMATCH` (dace `a` is all `1.0`).
  - `bisect_stages.py` names the first bad stage: `first bad stage index 52 distribute DistributeProducerConsumerLoop`.
- **Proposed minimal fix:** after the union-find pass, also merge every block whose position lies between a class's
  minimum and maximum member into that class, and repeat until nothing changes. This makes every group a contiguous
  interval, so the original order is preserved.
- **Out of my area, same program:** with this pass stubbed out, the program still miscompiles, and
  `bisect_stages_nodist.py` points to `first bad stage index 64 fission PerfectLoopNesting`. That pass likely has the
  same ordering flaw. Whoever owns it should check it.

---

## 5. ArgMaxLift (value + index path) drops the pre-loop seeds without checking them

- **Severity:** miscompile (frontend, full `canonicalize()`)
- **Files and lines:** `dace/transformation/passes/canonicalize/arg_max_lift.py`
  - line 712: `_verify_affine_seed` runs only when `folds_base`
  - `_rewrite_with_index` at 1790-1795: both carriers' pre-loop binds are dropped
- **Root cause.** `_verify_affine_seed`, which proves the seed is `a[start-1]`, runs only for the value-only and
  transform+index shapes. The index-only path (s315) is not checked. `_rewrite_with_index` still pops the pre-loop
  assignments of both the value carrier and the index carrier, and rebuilds the seed as the slice element at
  `start-1`. Any other seed is lost, for example `x = 2.0, idx = -1` where the seed is above every element, or an index
  seed that does not match the position of the value seed.
- **Reproducer:** `argmax_index_seed.py`
  ```
  library nodes: ['ArgReduce']
  expected [2.0, -1.0], got [0.9127555772777217, 5.0]
  MISMATCH
  ```
  With the lift stubbed out (`python argmax_index_seed.py disable`, and also via `argmax_probe_noop.py idx_high_seed`),
  the result is correct: `idx_high_seed: libnodes=[] loops=1 -> ok`. In `argmax_probe.py`, these shapes pass: s314 from
  1, s315 `>`/`>=`, the predicate index, and the constant or offset value-only seeds.
- **Proposed minimal fix:** run `_verify_affine_seed` on every symbol-carrier path that drops the pre-loop binds, not
  only when `folds_base`. That covers index-only and transform-only. Concretely, change the guard to
  `if not self._verify_affine_seed(...)`, since zero-base shapes also rebuild the seed positionally.

---

## Unverified suspicions

1. `dead_carried_store.py:162-169, 207-210`: the offset of a store or read is taken from `edge.data.subset`. On an
   AccessNode→AccessNode copy whose memlet names the other endpoint, that is the other array's subset, so the offset
   is misread. The frontend and trivial-tasklet shapes I tried did not produce such a memlet.
2. `assume_symbols_nonnegative.py:117-138`: `nonnegative=True` is stamped on every signed free symbol of every nested
   SDFG and on symbols that are not arguments, but the runtime guard traps only top-level argument symbols. A negative
   nested or derived symbol can then fold `CMod` into `Mod`, or `Min`/`Max`/`int_floor` incorrectly. I could not build
   a failing program.
3. `debug.py:228`: `canonicalize_with_stage_checks` runs `_build_stages()` outside the
   `symbolic.serialization_symbol_dtypes(authority)` context that `canonicalize()` sets up, and it skips the
   post-pipeline cleanup. Its per-stage verdicts can therefore differ from a real `canonicalize()` run.

## Reviewed, nothing verified

- `annotate_loop_kinds.py`: hints only.
- `collapse_noop_cast.py`
- `eliminate_trivial_tasklets.py`
- `empty_state_elimination.py`: its unsafe cases need unstructured control flow, which `RequireStructuredControlFlow`
  excludes in the pipeline. Two examples: a start state that has in-edges drops the out-edge assignments on the entry
  path, and a sibling `src -> succ` edge crashes with a duplicate edge after a partial mutation.
