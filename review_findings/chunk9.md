# chunk9 findings (tasklets.py, tile_access.py, tile_dims.py, vectorize_{cpu_multi_dim,gpu,multi_dim}.py, widen_accesses.py)

All reproducers are in `/tmp/claude-0/-home-user/1191ce82-d309-5728-89cd-bb12c458f12b/scratchpad/review/chunk9/`.
Run each with `source /home/user/.venv/bin/activate; PYTHONHASHSEED=0 python <file>` from that directory.
They drive the pipeline the way `tests/passes/vectorization/helpers/harness.py` does: `to_sdfg(simplify=True)`, then
`canonicalize`, then `VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa="SCALAR"))`.

---

## 1. REPLICATE classification ignores the dividend's coefficient and offset: `a[(i+1)//2]` and `a[(3*i)//2]` load wrong elements

- **Severity:** miscompile (wrong numbers, no warning)
- **Where:** `dace/transformation/passes/vectorization/utils/tile_access.py:799` (`_detect_replicate_factor`), used at
  `:999-1008` (the REPLICATE branch of `classify_tile_access`)
- **Root cause:** `_detect_replicate_factor` accepts `int_floor(c*i + c0, k)` for any affine dividend. It only checks
  that `_affine_coeff_for(dividend)` is not `None`. The REPLICATE branch then records `dim_strides=1` and
  `replicate=k` and throws away `c` and `c0` (`dim_offset=None`). The load lowers this as the "contracted box":
  a base at `(c*i + c0)//k`, and lane `l` reads `base + l//k`. That is correct only when `c == 1` and `c0 % k == 0`.
  When `W % k == 0`, the phase-aware fallback in `tile_load.py` (`_phase_aware_lane_exprs`) is skipped. So
  `(i+1)//2` loses its phase, and `(3*i)//2` loses its stride.
- **Reproducer:** `repro_replicate_offset.py` (N=32, divisible, so no remainder or mask is involved)
  ```
  int_floor(i + 1, 2) -> (<PerDimKind.REPLICATE: 'replicate'>,) dim_strides (1,) replicate (2,)
  int_floor(3*i, 2) -> (<PerDimKind.REPLICATE: 'replicate'>,) dim_strides (1,) replicate (2,)
  repl_off match: False
    ref [0. 1. 1. 2. 2. 3. 3. 4. 4. 5. 5. 6. 6. 7. 7. 8.]
    vec [0. 0. 1. 1. 2. 2. 3. 3. 4. 4. 5. 5. 6. 6. 7. 7.]
  repl_coef match: False
    ref [ 0.  1.  3.  4.  6.  7.  9. 10. 12. 13. 15. 16. 18. 19. 21. 22.]
    vec [ 0.  0.  1.  1.  2.  2.  3.  3. 12. 12. 13. 13. 14. 14. 15. 15.]
  ```
  The plain `a[i // 2]` still works, in K=1 and in K=2. `a[(N-1-i)//2]` (coefficient -1) takes the same broken
  path.
- **Proposed fix:** in `_detect_replicate_factor`, return `k` only when the dividend's coefficient of `var_name` is
  exactly 1 and the constant term is provably `≡ 0 (mod k)`. Otherwise return `None`. The dim then falls to the
  existing GATHER / per-lane-expression path, which handles arbitrary functions of the iter-var the same way as the
  Mod case. An alternative is to always use `_phase_aware_lane_exprs` when `c != 1 or c0 % k != 0`. That still
  needs the stride `c` to be carried.

---

## 2. A lane-dependent multi-element body buffer raises `NotImplementedError` out of the pipeline and leaves the caller's SDFG invalid

- **Severity:** crash (and the SDFG is left half-transformed)
- **Where:** `dace/transformation/passes/vectorization/widen_accesses.py:712` (`_unwidenable_lane_dep_error` builds
  a `NotImplementedError`), raised at `:554` / `:585`. `VectorizeMultiDim.apply_pass` (`vectorize_multi_dim.py`)
  restores the snapshot only on `VectorizeUnsupported`.
- **Root cause:** `WidenAccesses.apply_pass` first widens the non-transient boundary memlets (step 2). Then
  `_propagate_lane_dep` (step 3) finds a per-lane transient that cannot be widened and raises
  `NotImplementedError`. The code calls this a "refusal", but only `VectorizeUnsupported` is caught by the
  orchestrator's snapshot/restore loop. So the exception escapes `VectorizeCPUMultiDim.apply_pass`, and the
  caller's SDFG keeps the step-2 widening. The input is an ordinary NumPy program: a 2-element scratch array per
  iteration.
- **Reproducer:** `repro_multi_elem_buffer_crash.py`
  ```
  EXCEPTION: NotImplementedError WidenAccesses: lane-dependent transient 'tmp' has non-scalar shape (2,); widening a multi-element per-lane buffer to (W, ...) is unsupported. Refusing rather th
  SDFG unchanged after failure: False
  SDFG INVALID after failure: InvalidSDFGEdgeError Dimensionality mismatch between src/dst subsets (at state block, edge a[_loop_it_0:_loop_it_0 + 8] -> [0] (a:None -> tmp:None))
  ```
- **Proposed fix:** make `_unwidenable_lane_dep_error` return a `VectorizeUnsupported` that names the enclosing map
  (`maps=(source_map_label(map_entry.map.label),)`, as the other refusals do). The orchestrator then restores the
  snapshot and tiles the remaining maps. Also, or instead, run the step-3 check before any step-2 mutation.

---

## 3. The per-lane symbol fanout writes the interstate RHS with `str()`, which emits `Subscript(idx, ...)`, so `a[idx[i] + 1]` does not compile

- **Severity:** crash (C++ compile error for a common gather-with-offset)
- **Where:** `dace/transformation/passes/vectorization/widen_accesses.py:153`
  (`emit_per_lane_symbol_fanout`: `iedge.data.assignments[plane] = str(rhs_sym.xreplace(repl))`)
- **Root cause:** sympy's `str()` prints a `dace.symbolic.Subscript` as `idx[...]` only when it is the top-level
  node. Inside any other expression (`idx[i] + 1`, `2*idx[i]`, ...) it prints the constructor form
  `Subscript(idx, Min(i + l, ub))`. That text goes verbatim into the interstate assignment and then into C++. A bare
  `a[idx[i]]` happens to print correctly, which is why the existing gather tests pass.
- **Reproducer:** `repro_fanout_subscript_str.py` (`b[i] = a[idx[i] + 1]`)
  ```
  validate: OK
  per-lane assignment: idx_slice_plus_1_lane0id_1 = Subscript(idx, Min(N - 1, _loop_it_0 + 1)) + 1
  ...
  COMPILE/RUN FAILURE: CompilationError ['.../gather_plus_one.cpp:33:35: error: ‘Subscript’ was not declared in this scope', ...]
  ```
  The `k = idx[i] + 1; b[i] = a[k]` form fails the same way.
- **Proposed fix:** render the RHS with DaCe's symbolic printer instead of `str()`:
  `iedge.data.assignments[plane] = symbolic.symstr(rhs_sym.xreplace(repl))`. Checked:
  `symstr(pystr_to_symbolic('idx[i] + 1'))` gives `(idx[i] + 1)`.

---

## 4. An index computed by a tasklet from a gathered value (`k = idx[i]*2; a[k]`) is classified CONSTANT

- **Severity:** crash (compile error; the vectorized SDFG passes `validate()`)
- **Where:** `dace/transformation/passes/vectorization/utils/tile_access.py:370`
  (`build_symbol_definition_map`, source 2, renames an input connector to the bare data name and drops its subset).
  Also `:644` (`_is_tile_dependent` follows interstate edges only) and `:984` (the "no direct tile var → BROADCAST"
  stop).
- **Root cause:** for `k = __t0 * 2` with `__t0 = int64(idx[i])`, the scalar-definition scan rewrites the connector
  `__in1` to the symbol `idx`. The resolved index for `a[__sym_k]` is then `idx*2`, which has no `Subscript` and no
  tile iter-var. `classify_tile_access` marks the dim CONSTANT, because `_is_tile_dependent` never looks at
  tasklet-defined scalars. The `a` reads therefore become lane-invariant scalar copies (`a_const = a[__sym_k]`).
  Meanwhile the producer of `k` is tiled, so the interstate `__sym_k = k` reads an `int64_t[8]`. The
  `staged_lane_dependent_index` guard catches only AN→AN copies, not a tasklet in between.
- **Reproducer:** `repro_tasklet_index_constant.py`
  ```
  interstate: [{'__sym_k': 'k', 'k_plus_1': 'k + 1'}]
    a[k_plus_1] classified ['CONSTANT']
    a[__sym_k] classified ['CONSTANT']
  validate: OK
  COMPILE/RUN FAILURE: CompilationError ['.../computed_gather.cpp:37:15: error: invalid conversion from ‘int64_t*’ {aka ‘long int*’} to ‘int64_t’ {aka ‘long int’} [-fpermissive]']
  ```
- **Proposed fix:** in source 2, when the connector's memlet subset is not a constant single element of the source,
  either rename the connector to the full `Subscript(data, subset...)` expression or add the scalar to
  `unreadable_writes`. Then extend `_is_tile_dependent` (or the Stop-2 check) to treat a symbol whose definition
  reaches a lane-dependent scalar as tile-dependent, which routes the dim to GATHER. A minimal alternative is to
  widen `staged_lane_dependent_index` to raise `VectorizeUnsupported` when a promoted index is written by any
  producer with a lane-dependent input.

---

## 5. `_resolve_body_nsdfg_symbol_aliases` breaks a rename cycle or chain in `symbol_mapping` (`{i: j, j: i}`), leaving "Missing symbols on nested SDFG"

- **Severity:** crash (invalid SDFG out of the public `normalize_loop_nests`). I could not reach it through the full
  `VectorizeCPUMultiDim` run: there, an earlier prep or inline step flattens this body first.
- **Where:** `dace/transformation/passes/vectorization/vectorize_multi_dim.py:890-904`
- **Root cause:** each bare-symbol mapping entry is treated as an independent rename. `replace_dict` renames the
  inner references simultaneously (correct). The per-entry loop then does `symbol_mapping[outer] = outer` and
  `remove_symbol(inner)` one entry at a time. For a swap `{i: j, j: i}`, handling `i→j` sets `mapping['j'] = j`,
  which overwrites the pending `j→i` entry. Handling `j→i` then pops `'j'` and removes the symbol `j`, which the
  body now references (the former `i`). A chain `{a: b, b: c}` breaks the same way.
- **Reproducer:** `repro_symbol_alias_swap.py` (hand-built map over (i, j) around a non-inlined RMW body NSDFG with
  `symbol_mapping={'i': 'j', 'j': 'i'}`)
  ```
  before: mapping {'i': j, 'j': i, 'N': N} inner symbols ['N', 'i', 'j']
  after:  mapping {'N': N, 'i': i} inner symbols ['N', 'i']
  inner memlets: ['ia[j, i]', 'ib[j, i]']
  INVALID: InvalidSDFGNodeError Node validation failed: Missing symbols on nested SDFG: ['j'] (at state main, node inner)
  ```
- **Proposed fix:** first compute the final identity set, `targets = set(renames.values())`. Then remove only the
  inner symbols in `renames.keys() - targets`, pop only mapping keys that are not also targets, and set
  `symbol_mapping[t] = t` for every target after the loop. Alternatively, skip the whole rewrite when any rename
  target is also a rename source (a cycle or chain).

---

## Unverified suspicions

- `target_isa='CUDA_WARP'` is documented as "implies device=GPU", but `VectorizeMultiDim.__init__` only maps
  `"CUDA"` to the GPU device (`is_gpu_device`, `self._device`). With `CUDA_WARP`, `_device` is CPU,
  `MarkTileDims(require_gpu_resident=False)` tiles host maps, and the nodes are stamped `implementation='cuda_warp'`
  on a host SDFG (verified structurally). Compiling it was not tried.
- `VectorizeGPUMultiDim` forces `assume_even=True` for every non-branched request, including K>=2 with an explicit
  `masked_tail`. A non-divisible symbolic extent then relies on the runtime assume-even abort rather than a remainder.
  This is intended per the docstring, but it silently overrides an explicit caller choice. Not run: no GPU.
- `restore_sdfg_in_place` after a refusal leaves the SDFG identical except for node and memlet `guid`s, which are
  regenerated by the deep copy (verified). If a first refusal marks maps and a later one names none, the restored
  snapshot keeps the `NO_VECTORIZE` label suffixes while `apply_pass` returns `None`. Not constructed.
