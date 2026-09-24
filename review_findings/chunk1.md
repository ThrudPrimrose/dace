# chunk1 findings (canonicalize: finalize, forward_store_to_load, fuse_chained_scalar_reductions, fuse_conditions,
# fuse_consecutive_loops, fuse_loops, hoist_iv_updates, hoist_loop_range_calls, induction_variable_substitution,
# lift_inv, lift_loop_carried_reduction)

Checkout: /home/user/dace @ 4ea75f9. All reproducers are in
`/tmp/claude-0/-home-user/1191ce82-d309-5728-89cd-bb12c458f12b/scratchpad/review/chunk1/` (called `R/` below). Run them as
`cd R && source /home/user/.venv/bin/activate && PYTHONHASHSEED=0 python <file>`. The `*_isolate.py` scripts run the real
recipe (`pipeline._build_stages()`) up to the pass's position and then apply only that pass. Each one shows the output
before and after the pass, so the blame lands on that pass.

---

## 1. FuseConsecutiveLoops fuses `[A,B)` and `[B,C)` into `[A,C)` without proving `A <= B <= C`

- Severity: miscompile (full pipeline)
- Location: `dace/transformation/passes/canonicalize/fuse_consecutive_loops.py:211` (`_adjacent_identical`), merge at
  `_merge` (lines ~269-282)
- Root cause: the pass only checks that the first loop's exclusive end equals the second loop's start. It never checks
  that either range is non-empty in the right direction. If `B > C`, the original second loop runs zero times, but the
  fused loop stops at `C` and drops iterations `[C, B)` of the first loop. If `A > B`, the fused loop runs iterations that
  neither original loop ran. Both are legal inputs, because nonnegative symbols do not order `M` against `K`.
- Reproducers:
  - `R/fcl_nonadjacent.py` applies the pass directly to the frontend SDFG. Output:
    `pass result: 1` / `loop i i = 0 | (i < K)` / `reference 15.0 got 6.0`
  - `R/fcl_nonadjacent_pipeline.py` runs full `canonicalize()`. Output:
    `M,K = (5, 3) reference 15.0 got 6.0` / `M,K = (3, 5) reference 15.0 got 15.0`
- Proposed fix: in `_adjacent_identical`, require
  `symbolic.provably_nonnegative(B - A, assume_symbols_nonnegative=True)` and the same for `C - B` (B = the adjacency
  point, C = the second loop's exclusive end). Refuse otherwise. The tile/remainder shape
  (`0`, `int_floor(N,K)*K`, `N`) must still pass this check; if the prover cannot show it, add a floor-aware check. Apply
  the same gate in `plan_guarded_fusion`, which has the same adjacency-only test.

## 2. FuseConsecutiveLoops' body signature ignores the contents of NestedSDFG, Map and LibraryNode nodes

- Severity: miscompile (hand-built SDFG, compiled)
- Location: `fuse_consecutive_loops.py:114` (`_node_key` returns `('other', type(node).__name__)`)
- Root cause: two loop bodies count as "identical" when their node keys and edge memlets match. For any node that is
  not an AccessNode or a Tasklet, the key is just the class name. The key does not include the nested SDFG, the map
  range or schedule, or the library node's properties (for example Reduce `wcr`/`axes`). Two bodies that call different
  nested SDFGs through identical memlets therefore compare equal. `_merge` then deletes the second body and runs the
  first body over the union range.
- Reproducer: `R/fcl_nested_body.py`. Loop 1 over `[0,M)` calls a nested SDFG computing `x*2`. Loop 2 over `[M,10)`
  calls one computing `x+100`. Output:
  `pass result: 1` / `loops left: ['first']` /
  `reference [  0.   2.   4.   6. 104. 105. 106. 107. 108. 109.]` / `got [ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18.]`
- Proposed fix: refuse any body with a node that is not an AccessNode or Tasklet (the documented target shape is a
  re-rolled scalar body). The alternative is a real structural key, for example a recursive signature of the nested SDFG
  plus the map range and properties.

## 3. FuseChainedScalarReductions checks only the top-level operator, so it re-associates non-reductions

- Severity: miscompile (full pipeline)
- Location: `fuse_chained_scalar_reductions.py:70` (`_binop_op`), used by `_collect_chains`
- Root cause: a step is accepted if the tasklet's RHS is *some* `BinOp` whose top-level operator is `+` or `*`. The pass
  never checks that both operands are the bare accumulator connector and the bare increment connector.
  `o = acc * 0.5 + inc` (a damped linear recurrence) therefore passes as `acc + inc`. The fold rewires the first step's
  "increment" input to `incA + incB`, and the first tasklet keeps its `acc * 0.5 + ...` body, giving
  `acc*0.5 + (a + b)` in place of `(acc*0.5 + a)*0.5 + b`. The same flaw lets `o = acc + 2*inc` drop the `2*` of every
  later step.
- Reproducer: `R/fcsr_nonreduction.py` (explicit `with dace.tasklet` frontend program). Output:
  `direct pass result: 1` / `direct:   reference 0.8501941177580298 got 2.0747466324271344` /
  `pipeline: reference 0.8501941177580298 got 2.0747466324271344`
- Proposed fix: in `_binop_op`, or at the step match, require `rhs.left` and `rhs.right` to both be `ast.Name`. Require
  one of them to be the connector of `acc_edge` and the other the connector of `inc_edge`.

## 4. LiftLoopCarriedReduction strips the accumulator from a partial result another consumer still reads

- Severity: miscompile (full pipeline)
- Location: `lift_loop_carried_reduction.py:184` (`_trace_to_reduction`, copy shape); `_apply_lift`
- Root cause: in the frontend copy shape `red_tasklet -> transient -> copy -> map_exit`, the pass checks only that the
  transient has one producer. It does not check that the copy tasklet is the transient's only consumer. When the partial
  `t = P[i,j] + X[idx,i,j]` also feeds `Q[i,j] = t`, the lift rewrites the reduction tasklet to `t = X[idx,i,j]` and
  puts the WCR only on the `P` path. `Q` then receives the bare increment. The purity gate (`_accumulator_reads`) looks
  only at reads of the accumulator array, never at readers of the reduction's output.
- Reproducers:
  - `R/llcr_shared_partial.py` runs full `canonicalize()`. Output: `P max err 0.0 Q max err 5.467155943088367`
  - `R/llcr_shared_partial_isolate.py` runs the recipe without the `end` stage, then only this pass. Output:
    `no lift      P max err 0.0 Q max err 0.0` / `LiftLoopCarriedReduction result: 1` /
    `tasklet '__out = __in2' ['t[0]']` / `lift applied P max err 0.0 Q max err 5.467155943088367`
- Proposed fix: in `_trace_to_reduction`, require `st.out_degree(transient) == 1`, with that edge going into the copy
  tasklet. In the direct shape, likewise require the reduction tasklet's output connector to have exactly one out-edge.

## 5. LiftLoopCarriedReduction leaves a dangling accumulator reference when the increment reads the accumulator

- Severity: crash (codegen `KeyError`); a miscompile when the name resolves to something else
- Location: `lift_loop_carried_reduction.py:325` (`_match_reduction`) and `:196` (`_increment_ast`)
- Root cause: `_match_reduction` refuses fan-out of the accumulator's map-entry connector to *other* tasklets. It does not
  refuse the same connector appearing inside the increment operand of the same tasklet (`o = p + p * x`). `_increment_ast`
  returns `p * x` verbatim, and the lift then removes the `p` in-connector and its edge. What remains is `o = (p * x)`
  with a WCR, and it reads an undefined `p`. That expression is also a recurrence, not a reduction.
- Reproducer: `R/llcr_acc_in_increment.py` (full `canonicalize(validate=True)` passes validation, then compile fails).
  Output: `tasklet 'o = (p * x)' in ['x'] ['P[i, j] (CR: Sum)']` / `KeyError: 'p'`
- Proposed fix: refuse the candidate if `acc_conn` occurs as an `ast.Name` anywhere in the increment AST, meaning any
  operand other than the bare accumulator one.

## 6. InductionVariableSubstitution: closed-form trip count goes negative when the loop runs zero times

- Severity: miscompile (full pipeline)
- Location: `induction_variable_substitution.py:288` (`_extract_iv`), `:670` (`apply_use_site_substitution`), `:1261`
  (`_try_substitute_iedge_iv`)
- Root cause: all three sites use `trip_count = int_floor(end - start, stride) + 1`, where `end` is the inclusive last
  value. For `range(M, N)` with `M > N`, the loop runs 0 times, but the formula gives `N - M < 0`. The closed forms
  `init + c*trip` and `init * c**trip` then move the accumulator backwards (subtract, or divide).
- Reproducers:
  - Data accumulator (`_try_substitute`): `R/ivs_negative_trip.py`. Output:
    `pipeline: expected s=10 p=10, got s = 6.0 p = 2.5`.
    Isolated: `R/ivs_isolate.py`. Output: `loops before IVS: 2` / `IVS result: 2` / `loops after IVS: 0` /
    `isolated: expected s=10 p=10, got s = 6.0 p = 2.5`
  - Counter symbol (`_try_substitute_iedge_iv`): `R/ivs_negative_trip_iedge.py`. Output:
    `canonicalized: out = 98 (expected 100)`.
    Isolated: `R/ivs_iedge_isolate.py`. Output: `no IVS      out = 100` / `IVS result: 1` /
    `assignments {'k': '(-M + N + k)'}` / `IVS applied out = 98 (expected 100)`
- Proposed fix: use `Max(0, int_floor(end - start, stride) + 1)` at all three sites (sympy `Max` is emitted fine), or
  refuse unless `provably_nonnegative(end - start + stride)`. Factor the computation into one helper so the three sites
  cannot drift apart.

## 7. InductionVariableSubstitution (derived symbol): post-loop value is written even when the loop runs zero times

- Severity: miscompile (full pipeline; triggers at plain `N = 0`)
- Location: `induction_variable_substitution.py:1004` (`_try_substitute_derived_symbol`)
- Root cause: after inlining `sym := f(i)`, the pass materialises `sym := f(end)` on an unconditional edge after the loop.
  If the loop never executes, `sym` must keep its entry value. `f(end)` is instead `f(start - 1)`, for example
  `k := N == 0` for `k = i + 1` over `range(N)`.
- Reproducers: `R/ivs_derived_zero_trip.py`. Output:
  `frontend only: N=0 out = 100  N=4 out = 4` / `canonicalized: N=0 out = 0 (expected 100)  N=4 out = 4 (expected 4)`.
  Isolated: `R/ivs_derived_isolate.py`. Output: `no IVS      N=0 out = 100` / `IVS result: 1` /
  `assignments {'k': 'N'}` / `IVS applied N=0 out = 0 (expected 100)`
- Proposed fix: guard the post assignment with the trip condition. Either emit
  `sym := f(end)` only on an edge conditioned on `trip >= 1`, plus an else path that leaves `sym` unchanged, or refuse
  unless `trip >= 1` is provable.

## 8. LoopCarriedRotationSubstitution peels an unguarded iteration from a loop with a symbolic trip count

- Severity: miscompile (full pipeline; triggers at plain `N = 0`)
- Location: `induction_variable_substitution.py:1940` (`try_substitute_rotation`; the trip check just above at
  ~1919-1922 only fires for constant trip counts)
- Root cause: `LoopPeeling` emits the first iteration unconditionally. The pass checks `trip >= 1` only when the trip
  count is a constant; with a symbolic `N` it assumes the loop runs at least once. At `N = 0` the peeled iteration
  executes anyway and writes `a[0]`. The comment says `peel_limit` "licenses" this assumption, but that makes a legal
  input (`N = 0`, nonnegative by contract) produce wrong output.
- Reproducers: `R/rotation_zero_trip.py` (full pipeline). Output:
  `N=0 canonicalized a[:5] = [ 3.5 -1.  -1.  -1.  -1. ]  expected [-1. -1. -1. -1. -1.]` (the `N=4` result is
  correct).
  Isolated: `R/rotation_isolate.py`. Output: `no rotation      N=0 a[:3] = [-1. -1. -1.]` /
  `LoopCarriedRotationSubstitution result: 1` / `rotation applied N=0 a[:3] = [ 3.5 -1.  -1. ]`
- Proposed fix: refuse unless the trip count is provably `>= 1`, or wrap the peeled block in a `ConditionalBlock` on the
  loop condition evaluated at `start`. The same unguarded-peel assumption probably also affects `BestEffortLoopPeeling`
  (another file; not checked here).

## 9. HoistLoopRangeCalls never visits maps inside nested SDFGs

- Severity: crash (C++ compile error: `invalid increment expression`)
- Location: `hoist_loop_range_calls.py:73` (`for cfg in list(sdfg.all_states())`)
- Root cause: `SDFG.all_states()` recurses into control-flow regions but not into `NestedSDFG` nodes. A `CPU_Multicore`
  map with a call-bearing step (`int_ceil(N, 4)`) inside a nested SDFG therefore keeps the call in its OpenMP increment,
  which is exactly what this pass exists to remove. `_next_free_index` already walks `all_sdfgs_recursive()`, so only
  the traversal is missing.
- Reproducer: `R/hoist_nested.py`. Output: `pass result: None` / `call still in nested increment: True` / compile fails
  with `error: invalid increment expression` at `for (auto c = 0; c < N; c += int_ceil(N, 4))`.
- Proposed fix: `for sd in sdfg.all_sdfgs_recursive(): for state in list(sd.all_states()): ...`. `_bind_map_range`
  already uses `state.sdfg` and `state.parent_graph`, so it works unchanged per nested SDFG.

---

## Out of area, verified (for the owner of `materialize_loop_exit_symbols.py`)

- A counter stepped on a top-level edge and also inside a one-sided `if` in the same loop body gets the wrong exit
  value. In `R/ivs_cond_increment.py`, `k` after the loop is `6` instead of `36` under full `canonicalize()`, while
  `b[]` is correct. `R/cond_bisect2.py` binary-searches the flat recipe and blames
  `first bad unit: 45 reduce MaterializeLoopExitSymbols`. `R/ivs_cond_isolate.py` confirms that IVS itself refuses this
  shape (`IVS result: None`, numbers correct).

## Unverified suspicions

- `finalize.allocated_unconditionally` walks `parent_graph` only up to the nested SDFG's root. An access at the top of a
  nested SDFG that sits inside an outer `ConditionalBlock` therefore reads as "unconditional", and the buffer may be
  hoisted to `__dace_init` with a guard-dependent (possibly negative) extent.
- `HoistLoopRangeCalls._bind_map_range` excludes only the *parameters* of enclosing maps. A step naming a dynamic
  map-input connector (of the map itself or an enclosing one) would be hoisted onto an interstate edge where that name
  is undefined.
- `HoistInductionVariableUpdates.id_for` matches deep-copied nodes by `('access', data)`. A second, unconnected
  AccessNode of the IV's container in the same state would be removed from the residual body along with the component.
