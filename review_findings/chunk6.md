# chunk6 findings: convert_tasklets_to_tile_ops, demote_data_reading_interstate_symbols, enums, fuse_branched_tail_remainder, fuse_multiply_add, generate_tile_iteration_mask, insert_tile_load_store

Reproducers live in `scratchpad/review/chunk6/`. Every one drives the pipeline the way
`tests/passes/vectorization/helpers/harness.py` does: `to_sdfg(simplify=True)`, then `canonicalize`, then
`VectorizeCPUMultiDim(VectorizeConfig(widths=(8,), target_isa=SCALAR, remainder_strategy=...))`, then compile.
The result is compared against the un-vectorized reference compile (`drv.py`).
Run each one with `source /home/user/.venv/bin/activate; cd scratchpad/review/chunk6; PYTHONHASHSEED=0 python <file> [remainder]`.

---

## 1. A float literal meeting an int operand is lowered to a TileBinop that truncates the literal to int

- **Severity:** miscompile (default knobs, common source patterns, every remainder strategy)
- **Where:**
  - `dace/transformation/passes/vectorization/convert_tasklets_to_tile_ops.py:1973-2036` (`_convert_binop_with_symbol`)
  - `convert_tasklets_to_tile_ops.py:2158-2204` (`_convert_binop_with_two_symbols`)
  - The cast is emitted by `dace/libraries/tileops/nodes/tile_binop.py:257-271` (`_operand_dtype`).
- **Root cause:**
  - After `SplitTasklets`, `x = 0.25 * N`, `t = 0.5 * _loop_it_0` and `t1 = t0 * 0.5` (with `t0` an int scalar) become 0-in and 1-in tasklets.
  - The symbol converters turn them into `TileBinop`s with no dtype check. `_convert_binop` does have one, but it only covers two data connectors.
  - A lane-dependent symbol is also materialised as an **int64** lane tile.
  - The TileBinop expansion casts every Symbol operand to the dtype of the first data operand, or else to the symbol's declared dtype. The literal is therefore emitted as `(int)(0.25) * (int)(N)` and `(int64_t)0.5 * lane`, which is 0. The output descriptor is `double`, so nothing fails.
- **Reproducer:** `float_literal_times_int_operand.py` (argument: `scalar_postamble` | `masked_tail` | `full_mask`; all three fail).

  ```
  == scale_by_quarter_n  numpy: [0.     0.2625 0.525  0.7875]           # C[i] = A[i] * (N / 4)
  MISMATCH C: ... ref [0.2625 0.525  0.7875 1.05   1.3125] vec [0. 0. 0. 0. 0.]
  == shift_by_half_n_plus_one  numpy: [11.   11.05 11.1  11.15]        # C[i] = A[i] + (N + 1) * 0.5
  MISMATCH C: ... ref [11.   11.05 11.1  11.15 11.2 ] vec [0.5  0.55 0.6  0.65 0.7 ]
  == ramp_half  numpy: [0.   0.55 1.1  1.65]                           # C[i] = A[i] + i * 0.5
  MISMATCH C: ... ref [0.55 1.1  1.65 2.2  2.75] vec [0.05 0.1  0.15 0.2  0.25]
  == ramp_only  numpy: [0.  0.5 1.  1.5]                               # C[i] = i * 0.5
  MISMATCH C: ... ref [0.5 1.  1.5 2.  2.5] vec [0. 0. 0. 0. 0.]
  RESULT: WRONG
  ```

  The expanded tasklet (`ndiv4_dbg.py`) is `_c = ((int)(0.25) * (int)(N));`.
- **Proposed minimal fix:** in both symbol converters, compare the output descriptor dtype with the dtype the operands will be cast to (tile or scalar operand dtype, int64 for a materialised lane tile, or the symbol dtype) for non-comparison ops. If they differ:
  - leave a scalar-output tasklet as the Python tasklet it was (`return False`), because Python semantics are right;
  - raise `VectorizeUnsupported` for a tile output.

  The equivalent library-side fix is for `_operand_dtype` to use `out_dtype` for arithmetic (non-comparison, non-logical) ops.

## 2. `C[i] = A[off[0] + i]` splats lane 0 across the tile

- **Severity:** miscompile (every remainder strategy)
- **Where:** `dace/transformation/passes/vectorization/insert_tile_load_store.py:640-664`, the structured-read branch of `_stage_reads_in_state`. The upstream disagreement is in `widen_accesses.py` `_widen_subset_inplace`.
- **Root cause:**
  - The frontend hoists the index into a body interstate symbol, `off_slice_plus_i = _loop_it_0 + off[0]`. It is correctly not demoted, because it indexes a memlet.
  - `classify_tile_access` resolves the symbol's definition and reports the dim as LINEAR with stride 1.
  - `WidenAccesses._widen_subset_inplace` looks for the iter-var literally in the subset begin, finds none, and leaves the source memlet one element wide: `A[off_slice_plus_i]`.
  - InsertTileLoadStore then builds a stride-1 `src_kind='Tile'` TileLoad on that volume-1 memlet (`Memlet.from_memlet(s_edges[0].data)`). Codegen passes `_src` by value, and the by-value `tile_load` overload broadcasts it to all W lanes.
- **Reproducer:** `offset_from_array_read.py` (argument: `scalar_postamble` | `masked_tail` | `full_mask`; all fail).

  ```
  numpy expects [ 3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]
  MISMATCH C: first bad idx (array([1, 2, 3, 4, 5]),) ref [4. 5. 6. 7. 8.] vec [3. 3. 3. 3. 3.]
  RESULT: WRONG
  ```

  The expansion is `tile_load<double, 8, false>(_dst, _src, nullptr, (1) * (1))` with `_src` = `A[off_slice_plus_i]`.
- **Proposed minimal fix:** in `_stage_reads_in_state` (and in `stage_write_group` for writes), refuse with `VectorizeUnsupported` whenever the record gives a tile dim a non-CONSTANT kind but the staged memlet has size 1 in that source dim. The proper fix is for `_widen_subset_inplace` to find the dominating iter-var through the same `build_symbol_definition_map` the classifier uses, so the window becomes `A[off_slice_plus_i : off_slice_plus_i + 8]`.

## 3. FuseMultiplyAdd fuses integer `a*b + c` into `std::fma`, which computes in double

- **Severity:** miscompile (opt-in `VectorizeConfig.fuse_multiply_add=True`). The knob is documented as "up to one ULP", but for int64 the error is unbounded.
- **Where:** `dace/transformation/passes/vectorization/fuse_multiply_add.py:85-122` (`_fuse_in_state`). No dtype gate exists. It lowers via `convert_tasklets_to_tile_ops.py:1875-1924` to `TileFMA`, which emits `std::fma(a, b, c)`.
- **Root cause:** the pass matches any two-input `*` feeding a two-input `+` through a single-use transient, with no dtype check. For integer operands, `std::fma(long, long, long)` promotes to double: products above 2^53 lose bits, and int wrap-around semantics are lost too.
- **Reproducer:** `fma_int64.py` (`C[i] = A[i] * B[i] + A[i]`, int64, A = 300000001, B = 300000007).

  ```
  MISMATCH C: first bad idx (array([0, 1, 2, 3, 4]),) ref [90000002700000008 ...] vec [90000002700000000 ...]
  RESULT: WRONG
  ```
- **Proposed minimal fix:** in `_fuse_in_state`, skip the match unless the product transient, the addend and the output all share one floating-point dtype (`np.issubdtype(dtype.type, np.floating)`).

---

## Outside chunk6 (verified while probing, root cause not in these files)

- **`A[idx[i] + 1]` gather does not compile.**
  - Reproducer: `gather_index_plus_one.py`. It fails under `scalar_postamble`, `masked_tail` and `full_mask` with ``error: 'Subscript' was not declared in this scope``.
  - Cause: `widen_accesses.py` `emit_per_lane_symbol_fanout` writes `str()` of a sympy expression holding an array read into interstate assignments, e.g. `idx_slice_plus_1_lane0id_0 = Subscript(idx, Min(...)) + 1`. The gather itself is correctly done with tile ops, so these symbols are also dead.
- **Stack smash in a kernel the vectorizer reports as "tiled nothing".**
  - Reproducer: `inner_loop_sum_stack_smash.py` (a per-row `s += A[i, k]` over `range(cnt[0])`). It prints `*** stack smashing detected ***`.
  - Cause: the canonical `Reduce(axes=[0])` reads a 1-D NSDFG-local array. `_RunExpandNestedSDFGInputs` (`vectorize_multi_dim.py`) rewrites its input to the 2-D `A[_loop_it_1, 0:n]` but leaves `axes` unchanged. The Reduce then reduces the size-1 dim and expands to a copy of n doubles into the scalar `s`.

## Unverified suspicions

- `fuse_branched_tail_remainder.py:399-400`: `_populate_scalar_branch` re-points only the symbol_mapping KEY equal to the outer tiled param. A tail NSDFG whose inner name for the param differs, or which maps another symbol to an expression of it, would run every tail lane at the tile start. Today `NestInnermostMapBodyIntoNSDFG` maps identity, so I could not trigger it (GPU-only, no nvcc here). Substituting the param inside every mapping VALUE would be robust.
- `convert_tasklets_to_tile_ops.py:269-273`: `lane_dependent_through_interstate_assignment` flattens every assignment into one dict, so a symbol assigned on several edges keeps only the last RHS. An earlier lane-dependent definition would be missed.
