# Area: interchange (commit 4ea75f9)

Reproducers: `/tmp/claude-0/-home-user/1191ce82-d309-5728-89cd-bb12c458f12b/scratchpad/review/interchange/`
(run with `source /home/user/.venv/bin/activate; PYTHONHASHSEED=0 python <file>`; captured output in `out_<name>.txt`).
Line numbers are for `dace/transformation/interstate/map_loop_interchange.py` (MLI) and
`dace/transformation/interstate/subgraph_fission.py` (SF) at 4ea75f9.

## 1. MapLoopInterchange repeats the other dataflow in the map's state once per loop iteration
- Severity: miscompile
- Location: MLI:60-62 (`refusal`, the "state holds more than the map" check)
- Root cause: the check only rejects non-AccessNode top-level nodes. An AccessNode-to-AccessNode copy in the same
  state passes. After the move, the whole state sits inside the loop, so the copy runs T times instead of once.
  The simplifier produces this shape from plain frontend code (`tmp[:] = B` is fused into the map's state).
- Reproducer: `r1b_frontend.py` (frontend: `tmp[:] = B; for i in map: for t in range(T): B[i] += tmp[i]`)
  ```
  [('assign_13_4', ['AccessNode', 'AccessNode', 'MapEntry', 'MapExit', 'NestedSDFG', 'AccessNode'])]
  applied: 1
  reference: [4. 4. 4. 4.]
  interchanged: [8. 8. 8. 8.]
  ```
- Fix: require every edge of the state to touch the map entry or map exit, and every AccessNode to be adjacent
  only to the map. Equivalently, refuse if any edge has neither endpoint in `{entry, exit}`.

## 2. MapLoopInterchange misses a symbol carried from one loop iteration to the next
- Severity: miscompile
- Location: MLI:22-26 (`carried_across_iterations`)
- Root cause: only an assignment whose right-hand side reads an assigned name counts as carried (`s = s + 1`).
  In `st0: B += s --(s = t)--> st1`, iteration t reads the `s` that iteration t-1 assigned. The RHS `t` is not
  itself assigned, so the check passes. Outside the map, each loop iteration is a fresh nested-SDFG call, and
  `s` goes back to its mapped-in outer value.
- Reproducer: `r2_carried_symbol.py` (hand-built; outer s=100, T=4; expected 100+0+1+2=103)
  ```
  applied: 1
  reference: [103. 103. 103.]
  interchanged: [400. 400. 400.]
  ```
- Fix: treat as carried every symbol assigned inside the loop that is read on a path from the loop start before
  its assignment. The conservative version: refuse if any lhs assigned in the loop is in `body.symbol_mapping`,
  in `body.sdfg.symbols`, or read (in a state, an edge condition or an assignment RHS) upstream of its assigning
  edge.

## 3. MapLoopInterchange accepts a loop whose bound symbol the body reassigns
- Severity: crash (compile error). If it compiled, the trip count would be wrong.
- Location: MLI:67-71 (bound check)
- Root cause: a bound symbol is accepted when `symbol_mapping[name] == name`. The check never asks whether the
  loop body also assigns that symbol on an interstate edge. For `for (t=0; t<M; t++) { ...; M = 2 }`, the
  condition now reads the outer `M`, and the inner `M = 2` stays in the nested SDFG. That SDFG has lost `M` from
  its argument list, so the generated C++ assigns an undeclared `M`.
- Reproducer: `r3_bound_reassigned.py` (hand-built, M=5)
  ```
  applied: 1
  reference: [2. 2. 2.]
  ... bound_reassigned.cpp:17:5: error: 'M' was not declared in this scope   (CompilationError on the interchanged SDFG)
  ```
- Fix: in `refusal`, also refuse when a bound name (or the loop variable) is an lhs of any
  `loop.all_interstate_edges()` assignment.

## 4. MapLoopInterchange's loop variable overwrites an outer symbol assigned on an interstate edge
- Severity: miscompile
- Location: MLI:65 (`var in entry.map.params or var in outer.symbols or var in outer.arrays`)
- Root cause: a name the outer CFG assigns on an interstate edge (`t = 7`) is not in `outer.symbols`, so the
  name-clash check misses it. Once the loop moves into the outer SDFG, the loop's `t` overwrites the outer `t`,
  and a later state reads the loop's final value.
- Reproducer: `r4_loopvar_clobbers_outer_symbol.py` (hand-built: `init --(t=7)--> map/loop(t<3) --> after: C[0]=t`)
  ```
  t in outer.symbols: False
  applied: 1
  reference: (array([3., 3.]), array([7.]))
  interchanged: (array([3., 3.]), array([3.]))
  ```
- Fix: also refuse when `var` is assigned on any interstate edge of the outer SDFG, is the loop variable of any
  enclosing or other LoopRegion there, or is in `outer.free_symbols` / `outer.used_symbols(all_symbols=True)`.

## 5. MapLoopInterchange misses a transient carried through a node that is also written
- Severity: miscompile (reads uninitialized memory)
- Location: MLI:27-30 with `move_if_into_loop.upward_exposed_reads` (move_if_into_loop.py:160-173)
- Root cause: `upward_exposed_reads` is node-level. A read counts as exposed only if its AccessNode has no
  incoming write. It ignores subsets and dynamic writes, so reading element `(t+1)%2` from a node that just
  received a write to `t%2` passes. That element was written by the previous iteration. After the interchange,
  each iteration gets a freshly allocated transient.
- Reproducer: `r6_transient_carried_same_node.py` (hand-built double buffer, heap transient; expected 0+1+2=3)
  ```
  applied: 1
  reference: [3. 3.]
  interchanged: [1.43231882e-315 1.43231882e-315]    (garbage; value differs run to run)
  ```
- Fix: in `carried_across_iterations`, refuse any body transient that is both read and written in the loop,
  unless every read memlet's subset is covered by non-dynamic writes earlier in the same iteration. The simplest
  conservative version: refuse any transient that is read at all, unless the node it is read from has a single
  non-dynamic in-edge whose subset covers the read.

## 6. SubgraphFission: can_be_applied accepts an interstate assignment before the cut, and apply then raises with the SDFG half-rewritten
- Severity: crash (and the SDFG is left mutated)
- Location: SF:170-181 (`can_be_applied`), failure at SF:195 (`MapFission.apply_to`)
- Root cause: the comment claims MapFission judges the unsplit body at least as strictly as the split one. That
  is false. `nest_sdfg_subgraph` turns a symbol assigned in the first half (`k = 3`) into a
  symbol->scalar->symbol output: a new edge `k = __sym_out_k`, where `__sym_out_k` is written by the first nest.
  That nest reads the map-indexed input `a`, so MapFission's taint analysis refuses the split body. The same body
  with `k` undeclared in `body.symbols` instead dies earlier, inside `nest_sdfg_subgraph`, with a KeyError.
- Reproducer: `r5_sf_symbol_across_cut.py` (hand-built: `s0 --(k=3)--> s1(cut) --> s2: c = t + k`)
  ```
  apply raised: ValueError Transformation cannot be applied on the given subgraph ("can_be_applied" failed)
  SDFG changed although nothing applied: True
  ```
- Fix: in `can_be_applied`, refuse when any interstate edge of the body carries assignments (the edge after the
  cut is already required to have none). Better: decide on a deepcopy by running the two `nest_sdfg_subgraph`
  calls plus `drop_uninitialized_inputs` there, then ask `MapFission.can_be_applied_to` and
  `InlineMultistateSDFG.can_be_applied_to` on the copy.

## 7. SubgraphFission crashes on a strided map or offset accesses: the unconditional inline refuses
- Severity: crash (and the SDFG is left mutated)
- Location: SF:196 (`InlineMultistateSDFG.apply_to` is called unconditionally after MapFission)
- Root cause: after MapFission the two maps live inside the nested SDFG. The outer memlets of that nested SDFG
  are propagated, for example `a[1:N:2]`, and are not whole-array. `InlineMultistateSDFG.can_be_applied` requires
  full subsets (multistate_inline.py:214/235), so `apply_to` raises. The shipped test only covers `map[0:N]` with
  `a[i]`. A map `1:N` whose body reads `a[i - 1]` fails the same way (scratch `p8_sf_numeric.py`: both cuts raise).
- Reproducer: `r7_sf_strided_map.py` (the test's `long_body`, map range `1:N:2`)
  ```
  map range: 1:N:2
  cut after the first body block: BinOp_18
    MapFission.can_be_applied -> True on body blocks ['BinOp_18', 'for_19', 'assign_21_8']
    MapFission.can_be_applied -> True on body blocks ['nested_sdfg_parent', 'nested_sdfg_parent_0']
  apply raised: ValueError Transformation cannot be applied on the given subgraph ("can_be_applied" failed)
  SDFG changed although nothing applied: True
  ```
  (`dbg7_inline_spy.py` shows `InlineMultistateSDFG.can_be_applied -> False` at that point.)
- Fix: apply the inline only when `InlineMultistateSDFG.can_be_applied_to(...)` holds; the two maps are valid
  while still nested. Alternatively, fold that check into the deepcopy-based `can_be_applied` from finding 6.

## Checked, no finding
- `LoopFission.can_fission` never mutated the graph, and `LoopFission.fission`, `MoveIfIntoLoop.push` and
  `MoveLoopInvariantIfUp.hoist` returning False always left `to_json()` identical. Checked with `h_refusal_identity.py`
  over every region of 57 programs from loop_fission_test, move_if_into_loop(_and_map_symmetric)_test,
  interstate/move_loop_invariant_if_up_test, move_map_invariant_if_up_test and
  cond_component_fission_and_move_if_into_map_e2e_test, with simplify on and off: 132 can_fission, 132 fission,
  62 push and 132 hoist calls, 0 mutations, 0 invalid results. `match_guard` / `match_loop` are pure reads.
- MLI on a WCR-reduced output, a multi-parameter map and a step -2 loop: numerically equal (`p5_wcr.py`). MLI
  adds the loop variable only as a nested-SDFG symbol mapped from the outer loop variable, not as an outer SDFG
  symbol. Block `.sdfg` / parent references are correct after apply.
- SF with a 2-D unit-stride map: correct (`p7_sf_strided.py` with `[0:N, 0:N]`).

## Unverified suspicions
- SF `cut_block` takes the first top-level block with a matching label. Uniqueness of top-level labels is not
  enforced for blocks added with `add_node`, so the cut can land on the wrong block.
- MLI: a map state that is itself inside an outer LoopRegion with the same loop-variable name as the moved loop
  gives nested same-name loops. I could not build a frontend case.
