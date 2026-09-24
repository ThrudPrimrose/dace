# chunk5 findings (untile_loops, wavefront_polyhedron, wavefront_skew, branch_normalization, bypass_trivial_assign_tasklets, config)

Reproducers are in `/tmp/claude-0/-home-user/1191ce82-d309-5728-89cd-bb12c458f12b/scratchpad/review/chunk5/`.
Run each with `source /home/user/.venv/bin/activate; PYTHONHASHSEED=0 python <file>`. The wavefront ones also need `OMP_NUM_THREADS=1`, so the wrong answer comes from the schedule and not from a race.
Compiler warnings (the GCC 13 notice) are left out of the outputs below.

## 1. UntileLoops only audits memlet subsets. Tasklet code and branch conditions that use `i`/`ii` are rewritten wrongly
- Severity: miscompile. It reproduces through the full `canonicalize()`.
- Where: `dace/transformation/passes/canonicalize/untile_loops.py:485` (`body_bound_texts`) and `:512` (`_audit_combined_access`) are the audit. The rewrite is at `:984-988` (`inner.replace_dict`).
- Root cause: the safety audit only reads memlet bounds (`body_bound_texts` walks `st.edges()` memlets). The rewrite then runs `replace_dict` over the whole inner region, which also touches tasklet code, ConditionalBlock conditions and interstate edges.
  - Case A substitutes `ii -> k - i` and then `i -> 0`. A bare `i` in a tasklet becomes `0`, and a guard `ii == 0` becomes `k == 0`.
  - Case B leaves a bare `i` free after the defining loop is gone. `i` and `ii` then become SDFG free symbols, and `i` becomes a required program argument.
- Reproducers:
  - `untile_tasklet_sym.py` (pass alone):
    ```
    before tasklet: __out = i
    UntileLoops returned 1
    after tasklet: __out = 0
    ref [ 0.  0.  0.  0.  4.  4.  4.  4.  8.  8.  8.  8. 12. 12. 12. 12.]
    got [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    match False
    ```
  - `untile_tasklet_sym_canon.py` (the same kernel through `canonicalize(sdfg, validate=True)`) gives the same `got` of all zeros, `match False`.
  - `untile_branch_on_inner.py` (`if ii == 0: a[i+ii] = 1`):
    ```
    conditions after: ['((_untile_k_0 - 0) == 0)']
    ref [1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0.]
    got [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    ```
  - `untile_caseB_tasklet_sym.py` (case B, `a[ii] = i`):
    ```
    after tasklet: __out = i
    free symbols: ['i', 'ii'] arglist: ['a', 'i']
    ```
- Proposed fix: before the rewrite, audit every use in the inner region, not just memlets:
  - `inner.free_symbols`
  - tasklet code: `symbols_in_code` over each tasklet in `inner.all_states()`
  - ConditionalBlock / LoopRegion conditions
  - interstate edge conditions and assignments

  For case A, refuse if any non-memlet use of `i` or `ii` does not satisfy `depends_only_on_sum`. For case B, refuse if `i` appears anywhere. The simplest version: in both cases, refuse when `outer_var` or `inner_var` is used anywhere outside memlets.

## 2. UntileLoops' multi-dim ascent interchanges loops without a dependence check
- Severity: miscompile. It reproduces through the full `canonicalize()`.
- Where: `untile_loops.py:885` (`for candidate in _iter_candidate_inners(outer)`), `:168` (`_intermediate_chain_clean`), and the splice at `:990-1030`.
- Root cause: for `for ti: for tj: for i in range(ti, ti+K): for j in range(tj, tj+K)`, the pass pairs each tile loop with a same-axis point loop that sits deeper, past a foreign-axis loop. It then splices the point loop's body into that loop's parent.
  - The result is `(i, j)` row-major order. The program as written runs in `(ti, tj, i, j)` tile order.
  - This is a `tj <-> i` interchange. The only check is `_intermediate_chain_clean`, which looks at loop bounds, not data dependences.
  - A flow dependence of distance (1, -1) that crosses a tile column (`a[i+1, j] = f(a[i, j+1])`) is read before its write in tile order but after it in row-major order.
- Reproducers:
  - `untile_2d_interchange.py` (pass alone):
    ```
    UntileLoops returned 2
    max abs diff vs program: 0.3350595362952786 match False
    ```
  - `untile_2d_interchange_canon.py` (through `canonicalize(sdfg, validate=True)`):
    ```
    max abs diff vs program: 0.3350595362952786 match False
    ```
- Proposed fix: take a candidate below the immediate child (`candidate.parent_graph is not outer`) only if a legality check proves the interchange of the intermediate loops with the point loop is dependence-free. At minimum, require that no array is both written and read in the body with a subset that differs in the intermediate axis. Otherwise restrict the ascent to the immediate child (single-level only).

## 3. WavefrontSkew ignores every dependence carried by a container whose rank is not 2, then lifts with `proven=True`
- Severity: miscompile. It reproduces through the full `canonicalize()`.
- Where: `dace/transformation/passes/canonicalize/wavefront_skew.py:874` (`if desc is None or len(desc.shape) != 2: continue` in `scan_state_accesses`), and `:1751` (`parallelize_loop(sdfg, inner, proven=True)`).
- Root cause: the dependence collection skips scalars and 1-D/3-D arrays outright. The legality test only sees the single 2-D carrier. The tile-column (or front) loop is then turned into a Map with `proven=True`, which bypasses LoopToMap's own checks, and the tile/diagonal order replaces program order.
  - Here a 1-element accumulator `s[0] = 0.5*s[0] + a[i,j]` is threaded through a 2-D stencil. It is order-sensitive, and the skew reorders it.
- Reproducers:
  - `wavefront_ignores_1d.py` (pass alone, `tile_i = tile_j = 4`, N=12):
    ```
    WavefrontSkew returned 1
    [('_skew_t_0', True)]
    s ref [0.01692147] got [0.01752941]
    a max abs diff 0.0008147952990004148 match False
    ```
  - `wavefront_ignores_1d_canon.py` (N=150, default 64x64 tiles, `canonicalize(sdfg, validate=True)`):
    ```
    [('_skew_t_0', True, 'wavefront tile diagonal (t = _loop_it_0 + j) [64x64] -- ...')]
    s ref [1.09493677e-26] got [1.0973625e-26]
    a max abs diff 0.0004578539845531554 match False
    ```
- Proposed fix: in `scan_state_accesses`, refuse (`return False`) as soon as the inner body writes any container that is not 2-D, unless it is provably iteration-private. For example, a transient written before it is read in the same state and not accessed outside the nest. Treat reads from interstate-edge assignments and conditions inside the nest the same way.

## 4. BypassTrivialAssignTasklets reads subsets from the wrong side of a memlet
- Severity: miscompile.
- Where:
  - `dace/transformation/passes/vectorization/bypass_trivial_assign_tasklets.py:375-377` (dst branch: `de_subset = de.data.subset` is used as the consumer's `other_subset`)
  - `:308-312` (src branch: `pe.data.subset` is used as the producer's `subset`)
- Root cause: `Memlet.subset` belongs to `memlet.data`, which may name either end of an AccessNode-to-AccessNode edge.
  - In the dst branch, the standard copy memlet `T[1] -> [3]` has `data='T'`. So `de.data.subset` is T's index, yet it is installed as the consumer C's `other_subset`. The write lands at `C[1]` instead of `C[3]`.
  - The src branch has the mirror bug for a producer memlet that names its destination.
- Reproducers:
  - `bypass_other_subset.py` (dst branch, `A[5] -> [_out=_in] -> T[1] -> C[3]`):
    ```
    pass returned 1
    ref [  0.   0.   0. 105.   0.   0.   0.   0.   0.   0.]
    got [  0. 105.   0.   0.   0.   0.   0.   0.   0.   0.]
    match False
    ```
  - `bypass_other_subset_src.py` (src branch, `A[5] -> T[1]` memlet named on T):
    ```
    ref [  0.   0.   0. 105.   0.   0.   0.   0.   0.   0.]
    got [  0.   0.   0. 101.   0.   0.   0.   0.   0.   0.]
    ```
- Proposed fix: use `de.data.get_src_subset(de, istate)` / `get_dst_subset(de, istate)` and `pe.data.get_src_subset(pe, istate)`. Build the new memlet from the source-side subset of the kept endpoint and the destination-side subset of the other endpoint.

## 5. BypassTrivialAssignTasklets (src branch) re-points producers of other elements onto the copied element
- Severity: miscompile. Two writes race onto one element, and the wrong value wins.
- Where: `bypass_trivial_assign_tasklets.py:293-317`.
- Root cause: the src branch requires `out_degree(src_an) == 1` but puts no condition on the producers. Every in-edge of the transient is re-pointed at `dst[out_subset]`, including producers that wrote elements the copy never reads. `T[0] = 1; T[1] = 2; out[0] = T[0]` becomes `out[0] = 1; out[0] = 2`.
- Reproducer: `bypass_multi_producer.py`:
  ```
  pass returned 1
    inner edge: p1 -> out out[0]
    inner edge: p2 -> out out[0]
  ref [1.] got [2.] match False
  ```
- Proposed fix: take the src branch only when `src_an` has exactly one non-empty in-edge, and its destination subset equals `in_e`'s source subset (`in_e.data.get_src_subset(in_e, istate)`). Otherwise fall through to the dst branch or skip.

## 6. BypassTrivialAssignTasklets dedup merges copies of two different values of the same element
- Severity: miscompile.
- Where: `bypass_trivial_assign_tasklets.py:191-203` (`_dedup_identity_assigns`).
- Root cause: duplicates are keyed by `(src name, src subset, dst name, dst subset)` only. The key ignores which AccessNode (which version of the container) the copy reads.
  - `T = x[0]; x[0] = T + 1; T = x[0]; out[0] = 2*T` has two such copies, one before and one after the write to `x`.
  - They collapse into one. The second copy's consumer is rewired onto the first copy's `T`, which holds the stale value.
- Reproducer: `bypass_dedup_versions.py`:
  ```
  pass returned 1
  ref x, out = [11.] [22.]
  got x, out = [11.] [20.]
  match False
  ```
- Proposed fix: key on the AccessNode objects, `(id(in_e.src), subset, out_e.dst.data, subset)`. That way only copies out of the same source node, and therefore the same version, are merged. Alternatively, require `cur_src is kept_src`.

## 7. BranchNormalization serializes an if/else whose arm rebinds a guard symbol
- Severity: miscompile.
- Where: `dace/transformation/passes/vectorization/branch_normalization.py:507-525` (`guard_snapshot_verdict`), used by `serialize_two_arm` (`:642`) and `freeze_guards` (`:574`).
- Root cause: splitting `if c: A else: B` into `if c: A; if not c: B` is only sound if A does not change what `c` reads. The verdict only checks arrays (`arm_written_arrays`). An arm that reassigns a symbol the guard reads (interstate edge `k = k - 1` inside A) passes as "no snapshot needed", so the else half re-tests a mutated guard.
  - The cond resolver (`SameWriteSetIfElseToITECFG._resolve_cond_to_array`) also folds the arm's own `k = k - 1` binding into the lifted guard and stores it as `bool`. The generated C is `const bool _cond_k = (k - 1); ... (! (_cond_k > 0))`, which makes the result wrong even on the path where A did not run.
- Reproducer: `branchnorm_symbol_guard_serialize.py`. The SDFG is hand-built: `if k > 0: {x=1; k=k-1; y=1} else: {z=1}`, with `k` as an argument.
  ```
  original      [(k, x, y, z)] = [(0, 0.0, 0.0, 1.0), (1, 1.0, 1.0, 0.0), (2, 1.0, 1.0, 0.0), (3, 1.0, 1.0, 0.0)]
  BranchNormalization returned 2
  after pass    [(k, x, y, z)] = [(0, 0.0, 0.0, 0.0), (1, 1.0, 1.0, 0.0), (2, 1.0, 1.0, 1.0), (3, 1.0, 1.0, 0.0)]
  ```
  (Floats shown without the `np.float64(...)` wrapper.)
- Proposed fix: in `guard_snapshot_verdict`, return `None` (refuse serialization) when any interstate edge inside any arm of `cb` assigns a free symbol of the guard. Use `symbolic.symbols_in_code(cond_text)` intersected with the keys of `e.data.assignments` for `e` in each arm's `all_interstate_edges()`. Also never let the resolver inline a binding that sits inside `skip_cb`.

## 8. BranchNormalization changes the SDFG on a refused rewrite and reports "no change"
- Severity: perf-trap / pass-contract violation. It leaves stale analyses; not a wrong value.
- Where: `branch_normalization.py:217`. `_try_rewrite` calls `_hoist_branch_invariant_assignments(cb)` unconditionally, before the single-arm / two-arm checks, and ignores its return value.
- Root cause: the hoist deletes the arm's empty entry state and moves its binding onto `cb`'s in-edges, and the constant-binding sweep also moves bindings. When `_normalize_single_arm` then refuses (for example, the arm holds a Map), `_try_rewrite` returns `False` and `apply_pass` returns `None`, but the SDFG was restructured.
- Reproducer: `branchnorm_refusal_mutates.py`:
  ```
  BranchNormalization returned None
  states before: ['init', 'entry', 'body']  after: ['init', 'body']
  in-edge assignments after: [{'__sym_z1': 'z1'}]
  ConditionalBlock still present: True
  SDFG unchanged: False
  ```
- Proposed fix: return `True` from `_try_rewrite` (count it as a rewrite) when the hoist changed anything, making sure the fixpoint still terminates since a second hoist is a no-op. Or run the hoist only after the arm is known to be lowerable.

## Unverified suspicions
- WavefrontSkew: reads of the carrier inside the nest that go through interstate-edge assignments (`x = a[i-1, j]`) are not collected by `scan_state_accesses`. This is the same missing-dependence class as #3. Not reproduced separately.
- UntileLoops: after case A/B, the outer iterator `i` is no longer assigned. If code after the nest reads it (Python loop-exit value), `i` becomes free. Not reproduced through canonicalize, whose UniqueLoopIterators epilogue may cover it.
- wavefront_polyhedron.skew_bounds collects p-bounds across *all* basic sets of a coalesced set (max of lower bounds / min of upper bounds), which intersects a non-convex union instead of covering it. I could not build a domain that is non-convex after coalescing.
