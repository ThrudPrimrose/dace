# chunk4 findings (dace extended @ 4ea75f9)

Reproducers are in `scratchpad/review/chunk4/`. Run each with
`source /home/user/.venv/bin/activate; PYTHONHASHSEED=0 python <file>`.
Where a finding goes through `canonicalize`, `isolate.py`/`isolate2.py` wrap the pass's
`apply_pass` to capture the SDFG right before and right after it first fires. Both snapshots are
then compiled, so the wrong value is attributed to that pass alone.

## 1. RerollUnrolledLoops merges lanes whose tasklet reads the loop variable as a value
- Severity: miscompile (default CPU `canonicalize`, 'reroll' stage)
- Where: dace/transformation/passes/canonicalize/reroll_unrolled_loops.py:528 (the `_try_reroll` lane-identity check). The same gap exists in `_try_reroll_chain` (:754) and `_try_reroll_accumulator_reduction` (:895).
- Root cause: two lanes count as identical when their boundary array names and tasklet code strings match. Lane offsets are only read from memlet subsets and interstate RHSs. A tasklet that uses the loop variable `i` as a value (`__out = i`) has the same code in every lane, but lane k means `i` while the rerolled loop evaluates it at `i + k`. Nothing checks that the loop variable is absent from node code, map ranges or nested-SDFG symbol mappings.
- Repro: `rr1.py` (`for i in range(0,N,2): a[i]=i; a[i+1]=i`, full canonicalize, N=8). Control: `rr1_ctl.py` (reroll disabled).
  - `ref [0. 0. 2. 2. 4. 4. 6. 6.]`
  - `got [0. 1. 2. 3. 4. 5. 6. 7.]`, `match False`
  - with reroll disabled: `[0. 0. 2. 2. 4. 4. 6. 6.]`
- Fix: in all three matchers, refuse when the loop variable appears in the free symbols of any non-AccessNode body node (tasklet code, map range, `NestedSDFG.symbol_mapping`) or in an interstate edge that is not one of the recognised lane-symbol assignments.

## 2. RerollUnrolledLoops' new loop bound runs a zero-trip loop (truncating int_floor)
- Severity: miscompile, plus an out-of-bounds write when the arrays are sized to the loop
- Where: reroll_unrolled_loops.py:1020-1023 (`_rewrite_step`)
- Root cause: `last_i = init + step*int_floor(end - init, step)`, where `end = get_loop_end = bound - 1`. `dace::math::int_floor` is C truncating division (runtime/include/dace/math.h:99). When the original loop runs zero times, `end - init` is in `[-step, -1]`, so `int_floor` returns 0 instead of -1. The new bound `last_i + m*g` is then above `init` and the rerolled loop runs `m*g/g` iterations. The overlap case (`step < m*g`) is wrong even with floor semantics, because `last_i + m*g > init` there as well.
- Repro: `rr3_stage.py` (`for i in range(0, N-1, 2): a[i]=b[i]; a[i+1]=b[i+1]`, a/b sized N+2, b = 1..N+2)
  - `before loop: _loop_it_0 = 0 | (_loop_it_0 < (N - 1)) | _loop_it_0 = (_loop_it_0 + 2)`
  - `before N=1 [0. 0. 0.]`
  - `after loop: ... | (_loop_it_0 < ((2 * int_floor((N - 2), 2)) + 2)) | _loop_it_0 = (_loop_it_0 + 1)`
  - `after N=1 [1. 2. 0.]`, and `canonicalized N=1 [1. 2. 0.]`
  - The overlap variant is `rr2_stage.py` (3 lanes, step 2): `before 2 [0. 0.]`, `after 2 [1. 2.]`.
- Fix: derive the bound from a nonnegative trip count, e.g. `T = Max(0, int_ceil(bound - init, step))` and `new_excl = init + Piecewise((step*(T-1) + m*g, T > 0), (0, True))`. Alternatively, keep the original loop condition evaluated at `init` as an extra conjunct.

## 3. SplitStatements orders an ordered split by index delta, ignoring the loop variable's coefficient sign
- Severity: miscompile (default CPU `canonicalize`, 'prep' stage)
- Where: split_statements.py:488-514 (`access_offset`), used at :784-792 in `split_order`
- Root cause: `access_offset` returns the sign of `read_index - write_index`, and `split_order` reads a positive sign as "the read is ahead, so a LATER iteration writes it" and runs the reader first. That only holds when the index increases with `i`. With `a[N - i]` written and `a[N - i + 1]` read, the delta is +1, yet the element was written one iteration EARLIER (a true dependence). The pass therefore puts the reader loop first and it reads stale values.
- Repro: `ss1.py` (end-to-end) and `ss1_stage.py` (isolated). Loop: `for i in range(1,N): a[N-i]=x[i]; b[i]=a[N-i+1]`, N=8.
  - `SplitStatements fired: 1`
  - `before b [0. 0.5436 0.0027 0.8574 0.0336 0.7297 0.1757 0.8632 0. 0.]`
  - `after b [0. 0.5436 0.7295 0.6066 0.9128 0.8133 0.0165 0.041 0. 0.]`
- Fix: convert the delta into iteration space. Divide each dimension's difference by the loop variable's coefficient in the write index, which must be a nonzero integer constant, otherwise return `None`. Only then take the sign.

## 4. SplitStatements treats any store index that mentions `i` as one element per iteration (`iteration_distinct`)
- Severity: miscompile (default CPU `canonicalize`)
- Where: split_statements.py:665-676 (`iteration_distinct`), which gates `merge_carried_groups` and `split_order`'s same-index rule
- Root cause: the check is `loop_var in subset.free_symbols`. `s[i // 2]` passes it, but two iterations write the same element. The value is carried across iterations, yet the same-index rule sees `offset 0` with a post-write reader and orders the writer's loop first. The reader then sees each element's final value instead of its running value.
- Repro: `ss2_stage.py` (`for i: s[i//2] = s[i//2] + x[i]; b[i] = s[i//2]`, x = 1..8)
  - `SplitStatements fired: 1`
  - `ref b [ 1.  3.  3.  7.  5. 11.  7. 15.]`
  - `got b [ 3.  3.  7.  7. 11. 11. 15. 15.]`
  - before/after snapshots show the same flip
- Fix: require each store index to be affine in `loop_var` with a nonzero integer coefficient in at least one dimension (injective in `i`). Anything else (floor division, modulo, non-affine) counts as not iteration-distinct. The same helper then also fixes #3.

## 5. ShrinkMapLocalTransients resizes a transient that a NestedSDFG views with outer strides
- Severity: crash (Bus error, out-of-bounds access). Runs in `finalize` (finalize.py:323).
- Where: shrink_map_local_transients.py:212-224 (`apply_pass`). No candidate filter for edges into or out of a NestedSDFG.
- Root cause: `desc.set_shape(size)` recomputes the transient's strides, and the pass rewrites the outer memlets to the origin box. A NestedSDFG connected to the transient still describes its view with the old strides: a column slice `tmp[0:N, j]` is inner shape `[N]`, stride `N`, and after the shrink to `(N, 1)` the real stride is 1. The inner code indexes `k*N` into an `N`-element buffer.
- Repro: `smlt1.py` (map over j; NestedSDFG writes `tmp[0:N, j]`, another reads it). Control: `smlt1_ctl.py`.
  - `desc after: (N, 1) (1, 1)`
  - `n1 inner strides out/inp: (N,) (N,)`
  - then `Bus error` (exit 135)
  - control without the pass: `match True`
- Fix: skip candidates with any access edge whose src or dst is a NestedSDFG (or a View). Or, when shrinking, rewrite the nested descriptor's strides from the new outer strides for the accessed dimensions.

## 6. ReorderStateForLoopFusion misses dependences through interstate edges and symbols in loop2
- Severity: miscompile (GPU `canonicalize`, 'loop_fuse' stage; pipeline.py:1580)
- Where: reorder_state_for_loop_fusion.py:214-223 (`reorder_legal`)
- Root cause: legality uses `AccessSets[second]`, which for a LoopRegion rolls up only data-node accesses and its own header. It leaves out array reads on interstate edges inside the loop body (`k = x[i]`) and the conditions of nested conditional blocks. Symbols are never considered: moving `state` after `loop2` changes what it reads when `loop2`'s body or header assigns a symbol that `state` uses.
- Repro A: `rsf1.py`. `between` writes `x`; `loop2` reads `x` only through the body edge `k = x[i]`.
  - `order: [('s0','l1'), ('l1','l2'), ('l2','between'), ('between','end')]`
  - `ref  B [ 0.  5. 10. 15. 20. 25.]`
  - `pass B [0. 0. 0. 0. 0. 0.]`
- Repro B: `rsf2.py`. `between` reads symbol `k`; `loop2`'s body assigns `k = i*10`.
  - `ref  C [3]`, `pass C [50]`
- Fix: build `second`'s read/write sets from every block and interstate edge under it (`all_control_flow_blocks(recursive=True)` plus `all_interstate_edges(recursive=True)`, including conditional-branch conditions). Also refuse when `state`'s free symbols intersect anything `second` defines: its loop variable, interstate assignment LHSs, and symbols used by nested SDFGs.

## 7. sift_imperfect_nests moves a symbol assignment the inner loop's own bounds read
- Severity: miscompile (GPU only: PerfectLoopNesting target='gpu', perfect_loop_nesting.py:309)
- Where: sift_statements_into_perfect_nest.py:224-231 (`_match`); the rewrite is in `_sift`
- Root cause: the assignment on the `pre -> inner` edge (`k = 3*i`) is re-emitted inside the pre-guard in the inner body. The inner loop's init and condition (`j = k`, `j < k + 3`), and the guard `j == k` itself, are evaluated before that guard runs, so they see the previous outer iteration's `k`. `_match` only refuses when the BODY reassigns a sifted LHS. It never checks the inner loop's header. S1 (`k+2-k >= 0`) still proves the loop non-empty.
- Repro: `sift1.py` (called directly; semantics do not depend on the target)
  - `sifted 1`
  - `ref y [10. 11. 12. 23. 24. 25. 36. 37. 38. 49. 50. 51.]`
  - `got y [20. 21. 22. 33. 34. 35. 46. 47. 48. 49. 50. 51.]`
- Fix: refuse when any sifted interstate LHS (from both the pre->inner and inner->post edges), or any symbol/array the pre blocks write, is among the free symbols of the inner loop's `init_statement`, `loop_condition` or `update_statement`.

## 8. SinkStateIntoLoop ignores symbols (the pass is not wired into the pipeline)
- Severity: miscompile, but only for direct callers (no pipeline reference)
- Where: sink_state_into_loop.py:94-109 (`can_replicate`)
- Root cause: replicability is judged from data read/write sets only. A state that reads the second loop's loop variable, or a symbol its body assigns, sees a different value once it is sunk into the body.
- Repro: `sink1.py` (between-state `C[0] = i`, both loops over `i` in `0..4`)
  - `sunk 1`, `ref C [5] got C [4]`
- Fix: refuse when `state.free_symbols` intersects the loop variable or any interstate-assignment LHS inside `second`.

## Unverified suspicions
- reroll_unrolled_loops.py:109 `_assignment_offset` takes the index after the LAST `[`, so `ip[i+1] * c[i]` counts as offset 0.
- reconstruct_wavefront_nest.py:190 `_locate_loop` finds the probed loop by label. With duplicate labels the probe can test a different loop from the one committed. The pass is gated off by default.
- symbol_dedup.py: `id(edge)` keys and the sequential-assignment hazard are both safe, since validation rejects self-referencing same-edge assignments. Nothing found.

## Out-of-area observation
`rr2.py`/`rr2_ctl.py`: `for i in range(0, N-2, 2): a[i..i+2] = b[i..i+2]` gives `N=1 got [1.]`, `N=2 got [1. 2.]` (ref zeros) after full `canonicalize`, even with RerollUnrolledLoops disabled. Some other stage also mishandles this zero-trip loop. It is not attributed to any chunk4 pass.
