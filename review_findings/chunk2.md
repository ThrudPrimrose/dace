# chunk2 correctness findings

Reproducers live in `/tmp/claude-0/-home-user/1191ce82-d309-5728-89cd-bb12c458f12b/scratchpad/review/chunk2/`.
Run them from that directory:
`source /home/user/.venv/bin/activate; PYTHONHASHSEED=0 python <script> <mode> [variant]`.

Modes:
- `direct`: the frontend SDFG (`to_sdfg(simplify=True)` unless noted), then the pass on its own.
- `prefix`: `harness.run_until` runs the canonicalize recipe unit by unit and stops just before the pass, so the pass sees the same graph it gets inside the pipeline. Then the pass runs.
- `full`: `canonicalize(sdfg)` with default settings.

Findings 1-4 reproduce through the full default `canonicalize`. Findings 5-6 reproduce only when the pass runs on its own; in the default pipeline an earlier pass changes the shape first. See their notes.

---

## 1. LoopToConditionalReduce hoists every write in the guarded branch, not only the accumulator

- **Severity:** miscompile (reproduces through the full pipeline)
- **Where:** `dace/transformation/passes/canonicalize/loop_to_conditional_reduce.py:235-244` (legality check), `:482-497` (the hoist)
- **Root cause:** The matcher only checks the true-branch's *sink* access nodes: there must be one sink, and it must be the accumulator. The rewrite then moves the whole true-branch state out of the `ConditionalBlock` so it runs on every iteration. An access node in the middle of the accumulator chain is also a write, because it has an in-edge and an out-edge. That covers a non-transient argument (`c[0] = a[i]*2; s += c[0]`) and a transient that is read after the loop (`t = a[i]*2; s += t; ... b[1] = t`). After the hoist both of these are written even when the guard is false. The module docstring promises "No other writes (to non-transient arrays) inside the true-branch", but that is never checked for intermediate nodes.
- **Reproducer:** `cr2.py` (`mid_write_c` and `live_t`). `cr1.py mid_write` shows the same bug in direct mode on an array-indexed write `c[i]`.
  - `python cr2.py full mid_write_c` gives `got b [22.] c [-2.] expected 22.0 0.3999999999999999`. `c[0]` should keep the value from the last taken iteration; instead it holds the value from the last iteration of all.
  - `python cr2.py full live_t` gives `got [22. -2.] expected [np.float64(22.0), np.float64(0.3999999999999999)]`.
  - `prefix` mode reports `LoopToConditionalReduce: result=1 changed=True` and the same wrong values, which puts the fault on this pass.
- **Proposed fix:** In `_match`, refuse when any access node in `true_state`, other than the accumulator's own nodes, has `in_degree > 0` and either is non-transient or is read anywhere outside `true_state`. The "outside" check needs the same kind of whole-SDFG scan that `rank_k_match.internal_writes_contained` does. The simplest safe rule is to allow only transients whose every access node sits in `true_state`.

## 2. LoopToStreamCompaction accepts a nest whose inner trip count comes from an interstate symbol bound inside the outer loop

- **Severity:** miscompile (reproduces through the full pipeline)
- **Where:** `dace/transformation/passes/canonicalize/loop_to_stream_compaction.py:324-325` (`match_nest`, the "non-rectangular" gate)
- **Root cause:** The rectangularity gate only rejects an inner bound that names an outer *loop iterator*. `for k in range(cnt[i])` is lowered to `cnt_index = cnt[_loop_it_0]` on the outer body's interstate edge, and the inner loop's condition becomes `_loop_it_1 < cnt_index`. `cnt_index` is not an iterator, so the nest is accepted. The pass then shapes `mask` and `rank` as `[N, cnt_index]` and emits the Scan and Reduce outside the loop, where `cnt_index` has no per-row value. Rank and total are computed over the wrong extent, so both the scatter slots and the published cursor are wrong.
- **Reproducer:** `sc1.py ragged`
  - `python sc1.py full ragged` gives `has Scan: True`, `j got 5 expected 10 ; a ok: False`.
  - `python sc1.py prefix ragged` gives `LoopToStreamCompaction: result=1 changed=True`, `j got 5 expected 10 ; a ok: False`.
  - Loop bounds at the pass (from `sc1_inspect.py`): `for_11 ... (_loop_it_1 < cnt_index)` with `cnt_index` assigned on `for_10`'s body edge.
- **Proposed fix:** In `match_nest`, also reject a level whose `start` or `trip` names any symbol assigned on an interstate edge anywhere inside the outer levels. That means a symbol from `levels[k].loop.all_interstate_edges()` assignments for the enclosing levels, or more generally any symbol that is not defined at the root loop's position. A simple form: every free symbol of `start` and `trip` must be in `root.parent_graph`'s defined symbols at the point of `root`.

## 3. LoopToSymm (map form) lifts a symm nest over a partial map range to a full-array `Symm`

- **Severity:** miscompile (reproduces through the full pipeline)
- **Where:** `dace/transformation/passes/canonicalize/loop_to_symm.py:368-455` (`_match`), `:511-512` (`full()` memlets in `_replace`)
- **Root cause:** The map-form matcher looks only at subsets relative to the map parameters (`C[0:i, j]`, `C[i, j]`, `A[i, 0:i]`, ...). It never checks that the map's ranges are `0:shape` of `C`, `A` and `B`, or that the shapes agree (`A` is `M x M`, `B` and `C` are `M x N`). `_replace` always wires whole-array memlets. A map `j in 0:N-1` therefore becomes a `Symm` that also rewrites column `N-1` of `C`, which the original never touched. The slice form checks this with `shape_is(...)` against `loop_extent`; the map form has no equivalent check.
- **Reproducer:** `symm1.py`. It is the fixture from `tests/passes/canonicalize/loop_to_symm_test.py` with `j: _[0:N - 1]`.
  - `python symm1.py full` gives `Symm nodes: 1`, `ok: False ; last column untouched: False`.
  - `python symm1.py direct` gives `LoopToSymm: result=1 changed=True`, `ok: False ; last column untouched: False`.
- **Proposed fix:** In `_match`, require each map range to be `(0, extent-1, 1)`. Also require the `p_row` extent to equal `A.shape[0] == A.shape[1] == B.shape[0] == C.shape[0]`, and the `p_col` extent to equal `B.shape[1] == C.shape[1]`. Compare with `symbolic.inequal_symbols` / `equals`, and refuse otherwise.

## 4. LoopToSymmetrize and LoopToTranspose process nested-SDFG loops against the root SDFG's descriptors

- **Severity:** crash (invalid SDFG; reproduces through the full pipeline)
- **Where:** `loop_to_symmetrize.py:139-141` and `:176`; `loop_to_transpose.py:261-263`, `:315` and `:402-414`. `loop_to_einsum.py:1047-1050` has the same double loop.
- **Root cause:** `apply_pass` loops over `sdfg.all_sdfgs_recursive()` and, inside that, over `sd.all_control_flow_regions(recursive=True)`. The recursive walk already descends into nested SDFGs, so a nested SDFG's loops are first visited with `sd = root`. `_try_lift` then reads `sdfg.arrays[array]` from the root: same name, different descriptor. LoopToSymmetrize emits `X[0:6, 0:6]` memlets inside a nested SDFG whose `X` is `4x4`. LoopToTranspose calls `sdfg.add_view` on the root SDFG and uses the view inside the nested state. This is the documented `all_control_flow_regions(recursive=True)` double-visit gotcha. The first, wrong visit is the one that fires.
- **Reproducer:** `nest1.py` (`sym`, `tr`). An outer program passes `X[0:4, 0:4]` of a `6x6` array to an inner `@dace.program`; the nested SDFG survives to both passes.
  - `python nest1.py full sym` gives `libnode for_16_sym ['X_0[0:6, 0:6]', 'X_0[0:6, 0:6]']` and `InvalidSDFGEdgeError: Memlet subset out-of-bounds (at state sym_inner_23_call_sym_inner_23, edge X_0[0:6, 0:6] ...)`.
  - `python nest1.py full tr` gives `KeyError: 'A_tview'` inside `canonicalize`.
  - `prefix` mode gives the same errors, right after `LoopToSymmetrize: result=1` / `LoopToTranspose: result=1`.
- **Proposed fix:** Iterate `sd.all_control_flow_regions()` (non-recursive) inside the `all_sdfgs_recursive()` loop. Alternatively, keep the recursive walk alone and take the SDFG from `cfg.sdfg` / `outer.sdfg`. Apply the same fix to `LoopToEinsum._lift_loops`, which passes the root as `root` to `_match` and `_fold_coefficient` for nested loops.

## 5. LoopToEinsum's direct transpose matcher ignores the loop extents and emits a full-array `Transpose`

- **Severity:** miscompile (the pass on its own; not reached in the default pipeline, see note)
- **Where:** `dace/transformation/passes/canonicalize/loop_to_einsum.py:963-983` (`_direct_transpose`)
- **Root cause:** `_direct_transpose` checks the index permutation and the dtypes, then builds `full(sdesc)` / `full(ddesc)` from the descriptor shapes. It never compares the nest's axis ends with those shapes. For `for i in range(N): for j in range(N): B[i, j] = A[j, i]` on `M x M` arrays with `N < M`, it transposes all of `A` into all of `B`. The einsum branch uses `_axis_subset` (the loop extents) and the probe's `_extract_transpose` refuses partial ranges; this branch has neither guard.
- **Reproducer:** `ein1.py direct`, giving `LoopToEinsum: result=1 changed=True`, `libnodes: ['Transpose']`, `B ok: False`. Elements outside `[0:N, 0:N]` are overwritten.
- **Note:** In the default pipeline the copy tasklet is folded to an AccessNode-to-AccessNode copy before `loop_to_x`, which `_body_value` does not match (`ein1.py`/`ein2.py` in `prefix`/`full`/`full_nosem` modes are correct). LoopToTranspose also claims this shape first when `semantic_lifting` is on. So this is a latent legality bug in a pass that is also exported on its own.
- **Proposed fix:** In `_direct_transpose`, require each axis end to equal `shape[d] - 1` of the operand axis it indexes, for both src and dst. Refuse otherwise, or build the subsets from the axis ends, as `_axis_subset` does.

## 6. MaterializeLoopExitSymbols renames post-loop reads that follow a post-loop reassignment

- **Severity:** miscompile (the pass on its own; not reached in the default pipeline, see note)
- **Where:** `dace/transformation/passes/canonicalize/materialize_loop_exit_symbols.py:230-251` (`_rewrite_post_loop_readers`)
- **Root cause:** Every read of `sym` in every block reachable after the loop is renamed to `_loop_exit_<sym>_<n>`. The rename covers assignment right-hand sides but never the keys. After `k = k + 5` following the loop, the edge becomes `k = _loop_exit_k_0 + 5`. Every later reader of `k` is also renamed to the exit value, so it misses the `+ 5`. The rename has to stop at the first redefinition of `sym` along each path.
- **Reproducer:** `mx1.py direct` (loop `k = k + 2`, then `k = k + 5; b[0] = k; b[1] = k*3`) gives `MaterializeLoopExitSymbols: result=1 changed=True`, `b [12 36] expected b [17, 51]`.
- **Note:** In the pipeline, `IvSubstitutionFissionFixpoint` runs first and folds `k` away (`mx1.py prefix`/`full` are correct, and so are the `mx2`/`mx3` variants). A frontend-reachable shape that survives IVS was not found.
- **Proposed fix:** Rename only the reads reached before a reassignment. Walk forward from the loop's exit edges and stop propagating on any edge whose `assignments` contains `sym`. Rename that edge's right-hand side but not its successors. Or refuse when `sym` is assigned anywhere in `post_blocks`.

---

## Unverified suspicions
- `LoopToEinsum._lift_loops` (the same double visit as finding 4) computes `_fold_coefficient` (`beta`) from the root SDFG's same-named descriptor for a nested loop. That can choose beta=0/1 from the wrong transient flag or prior writer. Not reproduced.
- `LoopToConditionalReduce._collapse_empty_wrappers` (`:708`) assigns `loop.start_block` directly (gotcha G9). No failing case was built.
- Out of scope, seen while probing: an explicit `with dace.tasklet: b >> B(1, lambda x, y: x + y)[i, j]; b = a` inside a `for i, for j` nest fails to compile with no pass applied (`cannot convert 'double' to 'double*'`) and gives wrong values after full `canonicalize` (`tr1.py`). LoopToTranspose did not fire there.
