# chunk7 findings (vectorization passes, dace `extended` @ 4ea75f9)

Reproducers live in `/tmp/claude-0/-home-user/1191ce82-d309-5728-89cd-bb12c458f12b/scratchpad/review/chunk7/`.
Run with `./r.sh <script> [args]` (activates the venv, `PYTHONHASHSEED=0`, filters the GCC-13 warning). `drv.py` is the
shared driver: `to_sdfg(simplify)` -> `canonicalize` (optional) -> `VectorizeCPUMultiDim(VectorizeConfig(...))` -> compile ->
compare to numpy.

Four verified findings: 3 miscompiles you can hit through the pass entry points the pipeline uses, and 1 miscompile at
pass level that the current pipeline order happens to hide.

---

## F1. LowerITEToFpFactor blends with a non-0/1 condition: `c*t + (1-c)*e` is wrong for a truthy int guard

- **Severity:** miscompile. Reproduced through the full pipeline with `branch_mode="fp_factor"`.
- **Where:** `dace/transformation/passes/vectorization/lower_ite_to_fp_factor.py:56-58`
- **Root cause:** `_ITEToFpFactor.visit_Call` casts the ITE condition to the output dtype as-is,
  `cf = dace.int64(c)`, and treats it as a 0/1 factor. `SameWriteSetIfElseToITECFG` does not always
  produce a bool condition: its recipe 1 (the cond already names an array) and a value-typed
  `_lift_interstate_cond_to_tasklet` both keep the operand's own dtype. So `if b[1]:` with int64
  `b[1] == 3` reaches `ITE(_c, _t, _e)` with `_c = 3`, and the blend returns `3*10 + (1-3)*20 = -10`.
- **Reproducer:** `t_fpf_top.py` (`./r.sh t_fpf_top.py fp_factor`). The kernel runs a trivial
  map, then a top-level `if b[1]: d[0] = 10 else: d[0] = 20`.
  - Observed with `fp_factor`: `MISMATCH d / ref [10] / got [-10] / FAIL`.
  - Control: `./r.sh t_fpf_top.py merge` gives `MATCH`.
- **Proposed fix:** normalize the condition to a boolean before the cast:
  `cf = f"dace.{dtype}(({c}) != 0)"`. `bool` conditions keep the same result.
- **Side note (not a correctness bug):** `apply_pass` walks only `sdfg.all_states()`, which does not
  descend into nested SDFGs. An ITE inside a map-body NSDFG is therefore never blended. That is safe,
  but it means `fp_factor` is a no-op in exactly the case it exists for.

## F2. PrepareReductionForWidening: the in-state seed of `_priv_<acc>` has no ordering edge to the reduction map

- **Severity:** miscompile. The private accumulator is read uninitialized, and the seed runs after
  the map.
- **Where:** the entry point in this chunk is `dace/transformation/passes/vectorization/reduction_scalar_local_prep.py:88`.
  The defect is in the helper it delegates to,
  `dace/transformation/passes/canonicalize/privatize_reduction_accumulator.py:152-167` (the
  `in_state_init_an is not None` branch).
- **Root cause:** in `s[3] = 5.0; for i in map: s[3] += a[i]; c[0] = s[3]*2`, simplify puts the
  init, the map and the consumer in one state. The helper takes its in-state path:
  - It seeds a new `_priv_s` access node from the init node.
  - It points the map-exit WCR at a second `_priv_s` access node.
  - It adds no edge between the seed node and the map.
  
  Codegen orders the seed after the reduction loop (`copy_s_to__priv_s` is emitted last) and runs
  `reduction(+:_priv_s)` on an uninitialized scalar. The module docstring claims the seed and
  writeback go into separate states before and after the map, which is not what this branch does.
- **Reproducer:** `repro_prep_seed_order.py`
  - Simplified SDFG without the pass: `MATCH`.
  - After `PrepareReductionForWidening().apply_pass`: `MISMATCH s ref [... 14.82435014] got [... 9.82435014]`,
    `MISMATCH c ref [29.6487] got [19.6487]`, `FAIL`.
  - The full `VectorizeCPUMultiDim` on the un-canonicalized input gives the same `FAIL`. The
    orchestrator calls this pass unconditionally at `vectorize_multi_dim.py` (`PrepareReductionForWidening().apply_pass`).
  - Generated code: `.dacecache/rp_seed_prep/src/cpu/rp_seed_prep.cpp`.
- **Proposed fix:** in the in-state branch, order the seed before the map with
  `state.add_nedge(seed_an, state.entry_node(map_exit), Memlet())`. The alternative is to always take
  the separate init-state/writeback-state path.

## F3. LiftMapReductionToReduce (`rmw_only`) lifts a per-element update as a scalar reduction and hard-codes `acc[0]`

- **Severity:** miscompile.
- **Where:** `dace/transformation/passes/vectorization/utils/reductions.py:361-382` (`recognize_map_reduction`)
  and `dace/transformation/passes/vectorization/lift_map_reduction.py:597`.
- **Root cause:**
  - `recognize_map_reduction` accepts any one-element `map_entry -> body` read and `body -> map_exit`
    write of the same array. It never checks that the slot is independent of the map parameter, or
    that the read and write subsets are equal. So `x[i] = x[i] + a[i]` (NSDFG body, per-element
    connectors) is taken for a loop-carried scalar reduction whenever some tasklet in the state
    writes `0.0` into `x`.
  - `_lift` then writes the `Reduce` result to a hard-coded `f"{acc}[0]"`. The pure-WCR path's
    docstring explicitly warns against exactly that.
  - Result: `x[0] = sum(a)`, and every other element loses its `+ a[i]`. With a fixed slot `x[3]`,
    the sum lands in `x[0]` and `x[3]` keeps its stale value.
  - Minor: `for acc in set(reads) & set(writes)` iterates a string set, so which accumulator gets
    picked depends on the hash seed when several qualify.
- **Reproducers** (hand-built SDFG: init tasklet `x[0]=0.0`, then map `i` with an NSDFG body
  computing `xout = xin + ain` on `x[i]`, `a[i]`):
  - `t_rmw_indexed.py`: unlifted `MATCH`. After `LiftMapReductionToReduce(vectorized=True, rmw_only=True)`
    it prints `lift returned 1`, then
    `MISMATCH x ref [0.637 0.298 0.165 ...] got [9.824 0.028 0.124 ...]`, `FAIL`.
  - `t_rmw_slot3.py` (fixed slot `x[3]`): `lift returned 1`, then `ref x[3]=9.824, got x[0]=9.824 and x[3]=0.6706`, `FAIL`.
- **Proposed fix:** in `recognize_map_reduction`, require
  `str(read_edge.data.subset) == str(write_edge.data.subset)`, and require that neither subset has
  the map param among its free symbols. Iterate the candidates in a deterministic order (the
  `writes` dict order). In `_lift`, write `Memlet(data=acc, subset=copy.deepcopy(mx_out_edge.data.subset))`
  instead of `acc[0]`.

## F4. SameWriteSetIfElseToITECFG re-evaluates the lifted guard at the apply-ITE state instead of where it was defined

- **Severity:** miscompile at pass level. The current pipeline hides it for this shape, because
  `DemoteDataReadingInterstateSymbols` turns data-reading guard symbols into scalars first; the full
  `VectorizeCPUMultiDim` on this kernel gives `MATCH`.
- **Where:** `dace/transformation/passes/vectorization/same_write_set_if_else_to_ite_cfg.py`:
  - `_rewrite` at line 774 resolves the cond in `am_state`.
  - `_lift_array_predicate_cond` at line 1503 and `_lift_interstate_cond_to_tasklet` at line 1352
    inline the guard's defining RHS.
- **Root cause:** the guard symbol `a_index = a[i]` is assigned on the edge before `a[i] = b[i-1]`,
  and the block then branches on `a_index > 0.5`. The pass inlines the definition (`a[_loop_it_0] > 0.5`),
  re-reads `a` in the new `apply_ITE` state after the intervening write, and deletes the original
  assignment. Nothing checks that the RHS's arrays and symbols are unmodified between the defining
  edge and the block.
  - Same mechanism for `_lift_interstate_cond_to_tasklet`. It also takes the first assigning edge
    found anywhere, nested SDFGs included.
  - An NSDFG in/out connector pair aliasing the same outer element (`t_condmove_dump.py`) triggers
    it too.
- **Reproducer:** `repro_sw_cond_reeval.py`
  - Canonicalized SDFG: `MATCH`, with `guard symbol edge: block_0 -> slice_b_15 {'a_index': 'a[_loop_it_0]'}`.
  - After the pass: `lifted guard in state apply_ITE_if_16 : _out__cond_expr = (_in_a_0 > 0.5) ['a[_loop_it_0]']`,
    then `MISMATCH a`, `MISMATCH b` (b drifts to -17.58 vs ref 4.42), `FAIL`.
- **Proposed fix:** before inlining a symbol's RHS, require that no array it reads is written, and no
  symbol it reads is reassigned, on any path from its defining edge to `cb` (including `cb`'s arms).
  If that cannot be shown, stage the value into a transient scalar at the defining edge (the demotion
  the pipeline already does) instead of re-reading it at the merge. When refusing, fall back to
  keeping the symbol text.

---

## Out of this chunk, observed while building reproducers (canonicalize, not listed as known)

`canonicalize(sdfg)` alone breaks these kernels. The plain simplified SDFG is correct in each case.

- `t_twowcr_b.py`: `for i in map: s[0] += a[i]; s[0] += 2*b[i]`. After canonicalize, `s` stays at its
  initial value (`got [1.]`).
- `t_condsum4_full.py`: sequential loop over `i` containing a masked `s[i] += a[i,j]` inner map.
  Rows come out doubled, and which rows are doubled varies between runs, so it looks like a race.
- `t_condmove.py`: `c = a[i] > 0.5; a[i] = 0; if c: ...` inside a map. The `a[i]` write is hoisted
  ahead of the guard read.
- `t_slot3.py`: canonicalize-only also fails on the F2 kernel.

## Unverified suspicions

1. `lift_map_reduction._validate_pure_wcr_write` / `_lift_pure_wcr` never check `write_edge.data.dynamic`.
   A body that writes only conditionally would leave `_red_buf[j]` uninitialized, and `Reduce` would
   sum garbage. I could not build a frontend kernel that reaches it: canonicalize's `_nnr` seed makes
   the write unconditional.
2. `reduce_expansion._build_vectorized_full_reduction` indexes `__inp[_i + _l]` as a raw contiguous
   pointer, ignoring the squeezed input stride (a column slice). It also seeds all 8 lanes with
   `node.identity`, which counts a non-neutral identity 8 times. The lifts only emit contiguous
   buffers with a neutral identity, so this is unreached in the pipeline.
3. If-conversion in `SameWriteSetIfElseToITECFG` makes integer `//` / `%` in both arms unconditional,
   which could trap with SIGFPE for `if b != 0: q = a // b`. With `%`, `py_mod` did not fault here,
   and `//` hits an unrelated `__int_floor` compile error, so this stays unverified.
