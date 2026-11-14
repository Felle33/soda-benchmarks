# Resource-controlled FP32 GEMM lowering

Local implementation, 13 November 2026. Based on upstream
`6a355b18ac95011638938ee93dd50f71f4f74a66`; not published.

This opt-in path targets one matmul core with a declared arithmetic budget.
It retains the ordinary TOSA -> linalg -> LCS -> LLVM -> Bambu -> native ESP
flow. The workload requires no hand-written accelerator IR. It does not choose
fusion versus reuse, change numerical precision, or replace the existing outer
memory tiling model.

## Configuration and mechanism

Set `LCS_COMPUTE_LANES=64` when running `scripts/tosa_to_lcs.sh`. A small heuristic
chooses the exact factorization minimizing `rows + columns`, producing an 8x8
register block. This minimizes the operand values needed by an ideal outer
product step; it is an initial configuration, not a measured latency optimizer.
`LCS_REGISTER_BLOCK=R,C` overrides the block; `LCS_FP_UNITS` controls the HLS
resource limit. `LCS_BANK_LOCAL=false` disables the new bank mapping for studies.
With no compute option, the existing lowering is unchanged.

The inner block follows Tommaso's double-tiling study: keep R*C accumulators in
registers, advance k sequentially, load R values from A and C values from B, and
broadcast them across R*C multiply-adds. The compiler emits explicit scalar
lanes and SCF loop-carried accumulators. Guarded loads/stores handle partial
register blocks without dynamic lane-index cleanup loops. Reduction order and
separate FP32 multiply/add operations are preserved.

This supports tails of the **register block inside a local memory tile**.
It does not remove the existing outer-tiling/divisibility restrictions of the
new reusable-operator path or add arbitrary runtime shape support to that path.

Local memories are cyclically banked along A's row dimension and B/C's column
dimension. Buffer mappings propagate through the node call graph so the DMA
tasks and compute tasks share the same layout. Bambu requires divisibility by
the bank count, so physical dimensions are rounded up while logical subviews
keep original shapes, loop bounds and DMA lengths. Ping-pong ownership and
read/write admission remain intact.

The backend folds subview accesses and emits array-shaped LLVM allocations and
multi-index GEPs. It distinguishes whole-array allocas from C-style decayed
array arguments. Partition markers refer to these original physical buffers.
This avoids the flattened-address mismatch identified in the double-tiling study.

`compute_config.json` records the chosen block and units. `compute_storage.json`
reports local memref allocation bytes after padding. The existing `LCS_PLM_BYTES`
limit is checked again after padding; it is not silently increased. This byte
limit excludes HLS memory replication and FPGA BRAM granularity, which need
physical implementation checks.

The current resource-controlled pass accepts one static-local-buffer FP32
matmul per accelerator. It rejects multiple matmuls rather than accidentally
allocating the full budget to every stage of a fused computation. Alpha/beta
epilogue resources are reported separately. A global allocation across multiple
compute stages remains to be implemented.

## Selected toolchain and local integration fixes

The verified installation is `/home/tommaso/opt/bambu`, PandA 2026.11 revision
`cda4db2e724df357f749c0c41b12bdc065462ced-main`, with LLVM 19. The campaign mounts
it read-only. The installed binary and sources are not modified.

The complete banked dataflow exposed three issues beyond the standalone C
microbenchmark. Separate local copies contain these changes:

1. `ArrPart.cpp`: remove stale non-top interface entries after partitioning
   expands a node's parameter list.
2. `create_ir_manager.cpp`: read the architecture XML updated by compiler
   passes, instead of rereading the original user-supplied interface.
3. `plugin_customSROA.cpp`: recognize a control-flow merge whose incoming index
   expressions have identical ScalarEvolution expressions. Those indices have
   the same reachable banks. Differing/unknown expressions still conservatively
   fall back; recursion is bounded. This avoids spurious bank-pointer multiplexing
   in guarded compute lanes, which the dataflow interface rejects.

The campaign retains patch files, reproducible compile/link commands, source
hashes, original and patched binary/plugin hashes. The local driver is selected
with `BAMBU_BIN`; a read-only container mount selects the patched CSROA plugin.
The printed upstream version alone does not identify these local changes.

## Validation and evidence boundary

The local campaign is `experiments/gemm-resource-matched-20261113` in the outer
project workspace. Its manifests, rather than this prose, are authoritative.

- Four native shapes: 16x16x16, 10x25x30, 3x5x7 and 17x9x13; nonzero initial
  accumulators, two calls per case, exact dyadic reference values and padding guards.
- MINI component RTL: two launches without initial stalls and two with a
  50,000-cycle initial write stall; zero errors.
- MINI native ESP full-system RTL: 500 correct outputs, 17,137 accelerator cycles.
- MINI/LARGE HLS: 64 multipliers and 64 adders in the core, II=1, 20 ns target.
  The alpha/beta task has two additional multipliers and one adder.
- The budget-only entry point reproduces the qualified MINI/LARGE LLVM, XML
  and host IR byte for byte. Default lowering also matches its upstream control
  under the same toolchain.

The 2x2 VC707 LARGE implementation is tracked separately. HLS estimated area is
not FPGA LUT usage; requested clock/II is not a routed timing guarantee. MINI
latency must not be extrapolated into a LARGE or SPECTRA speedup. Existing
Experiment 1 FPGA results belong to their original artifacts and remain intact.

## Files to review

- `scripts/lcs/compute_budget.py`: bounded initial block selection.
- `scripts/tosa_to_lcs.sh`: optional block/bank passes and configuration records.
- `examples/soda-plugins/lib/sodap/Conversion/DataflowToLLVM/BankedMemrefs.cpp`:
  explicit compute lanes, padding/bank propagation, shaped LLVM conversion.
- The corresponding `Passes.td` and `CMakeLists.txt`: registration and build.
- `scripts/ll_to_verilog_esp_version.sh`: selected Bambu executable and FP limits.
- `examples/soda-plugins/test/Conversion`: numerical and padded-storage checks.

The mechanism is usable independently of a future search policy. Next tuning
should measure DMA service and achieved utilization before expanding the search
space or increasing compute resources.
