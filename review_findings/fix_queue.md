# Pending fixer launches (launch when a slot frees). Prompt: "Read FIX.md; worktree /home/user/wt/<wt>, branch <br>; findings X of findings/<file>.md"
- [x] PRIORITY NEXT: trim/merged-1 (launched) trim agent over merged integration branch (see task #7) — prompt in transcript
Trip-count family (int_floor truncation on empty/negative trip): base these on fix/stride-tripcount once it lands and reuse its helper.

- [x] branchnorm (chunk5 7,8) — launched
- [x] interchange (MLI 1-5, SF 6-7) — launched
- [x] f-cascade  fix/cascade-iedge-up        chunk0 #1 CascadeInterstateEdgeAssignmentsUp
- [ ] f-deadcarried fix/dead-carried-store  chunk0 #2,#3 DeadCarriedStoreElimination (trip-count family for #3)
- [ ] f-distribute fix/distribute-order     chunk0 #4 DistributeProducerConsumerLoop + PerfectLoopNesting fission ordering (distribute_frontend.py still wrong w/o distribute)
- [ ] f-argmax fix/argmax-seed              chunk0 #5 ArgMaxLift index-carrier seeds
- [x] f-fusecons fix/fuse-consecutive       chunk1 #1,#2 fuse_consecutive_loops
- [ ] f-fcsr fix/chained-scalar-red         chunk1 #3 fuse_chained_scalar_reductions
- [x] f-llcr fix/lift-loop-carried-red      chunk1 #4,#5 lift_loop_carried_reduction
- [ ] f-ivs fix/induction-var-subst         chunk1 #6,#7,#8 IVS (+ MaterializeLoopExitSymbols one-sided-if counter; BestEffortLoopPeeling unguarded peel check) (trip-count family)
- [ ] f-hoistrange fix/hoist-loop-range-nested chunk1 #9 hoist_loop_range_calls nested SDFGs
- [ ] f-reroll fix/reroll-unrolled          chunk4 #1,#2 RerollUnrolledLoops (trip-count family) + rr2_ctl zero-trip other stage
- [x] f-splitstmt fix/split-statements      chunk4 #3,#4 SplitStatements
- [ ] f-shrink fix/shrink-map-local         chunk4 #5 ShrinkMapLocalTransients
- [ ] f-reorder fix/reorder-state-loopfusion chunk4 #6 ReorderStateForLoopFusion
- [ ] f-sift fix/sift-imperfect             chunk4 #7 sift_imperfect_nests
- [ ] f-sink fix/sink-state-into-loop       chunk4 #8 SinkStateIntoLoop
- [ ] NF emit: 7 pre-existing e2e emit failures (Scan in map, non-inclusive Scan, MergeLibraryNode, azimint IndexError, scattering timeout) -> on top of debloat/ir after it lands
- [ ] f-stage fix/stage-global-array        chunk8 #1,#4 stage_global_array_through_scalars (dyn/IT write counted as overwrite; outer source fallback cycle)
- [x] f-mapred fix/recognize-map-reduction  chunk8 #2 utils/reductions.recognize_map_reduction accumulator-element check
- [ ] f-remainder fix/split-remainder-guard chunk8 #3 split_map_for_tile_remainder assume_even guard scope
- [ ] f-pow fix/power-expander-bound         chunk8 #5 PowerOperatorExpander unbounded unroll
- [x] f-condreduce fix/loop-to-conditional-reduce chunk2 #1 — launched
- [ ] f-compaction fix/stream-compaction-ragged chunk2 #2 loop_to_stream_compaction ragged inner bound
- [ ] f-symm fix/loop-to-symm-coverage      chunk2 #3 loop_to_symm map covers whole arrays
- [x] f-libnest fix/loop-to-lib-double-visit chunk2 #4,#5 loop_to_symmetrize/transpose/einsum double visit of nested SDFGs + einsum _direct_transpose bounds
- [ ] (fold into f-ivs) chunk2 #6 materialize_loop_exit_symbols renames reads after reassignment
- [ ] f-wcrcopy: chunk2 outside-note tr1.py WCR copy tasklet in for/for fails to compile / wrong after canonicalize
- [x] f-tileaccess fix/tile-access-index    chunk9 #1,#4 — launched
- [x] f-widen fix/widen-accesses             chunk9 #2,#3 widen_accesses NotImplementedError after mutation (orchestrator snapshot restore on any exception?) + symstr for fan-out subscript
- [ ] f-aliasswap fix/vectorize-symbol-alias-swap chunk9 #5 _resolve_body_nsdfg_symbol_aliases swap/chain renames
- [x] f-mapred also covers chunk7 F3 — launched
- [x] chunk7 F2 -> sent to f-postloop agent (privatize seed order)
- [x] f-canonmisc fix/canonicalize-misc: bisect t_twowcr_b, t_condsum4_full, t_condmove — launched
- [x] f-iteblend (merged into f-swcond) fix/lower-ite-fp-factor     chunk7 F1 LowerITEToFpFactor int cond not 0/1 -> (c)!=0
- [x] f-swcond fix/same-write-set-guard-reeval chunk7 F4 SameWriteSetIfElseToITECFG re-evaluates guard after overwrite (same file as branchnorm's lifter hunk: base on fix/branch-normalization when it lands)
DONE: fix/scatter 94b09f5, fix/tsvcvec 433251f, fix/branch-normalization f48a1e1
DONE: fix/offload d469bfc
- [x] f-widen also covers chunk6 #2 + gather_index_plus_one — launched
- [x] f-tilebinop fix/tile-binop-dtype chunk6 #1,#3 — launched
- [ ] f-reduceaxes fix/expand-nested-reduce-axes chunk6 outside: _RunExpandNestedSDFGInputs Reduce axes not remapped -> stack smash (inner_loop_sum_stack_smash.py)
ALL 11 REVIEWS DONE.
DONE+MERGED: fix/bypass-trivial-assign 4e45e570e (integration now +595/-44 before trim)
DONE+MERGED: fix/loop-to-lib-double-visit 9a25aef1c (after trim1 branch point: trim agent does not cover it)
