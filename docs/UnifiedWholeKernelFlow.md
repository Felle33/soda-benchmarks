# Unified kernel/network entry point

This local branch starts at `ba29c43`; the follow-up starts at `dfcdb4e`.
Phase 1 correctness/identity qualification and Phase 2 script cleanup are
complete. All 135 compiler tests pass. This is not a promoted FPGA campaign:
the new nonoverlap GEMM MEDIUM selection is **12.8% slower in component RTL**
despite a lower analytical DMA cost. No LARGE HLS, FPGA implementation or board
work was performed in the follow-up. Detailed evidence and the read-only serial
schedule diagnosis are in `../followup-20261117/REPORT.md` from this checkout.

## Entry points

Run in the pinned campaign container. Output directories must be new.

```bash
python3 scripts/pytorch_to_esp.py \
  examples/esp-validation/reusable-gemm/operators/polybench_models.py \
  --model-options '{"kernel":"gemm","dataset":"LARGE"}' \
  --mode kernel --schedule overlap --compute-lanes 64 \
  --plm-bytes 327680 --reuse-bytes 262144 --dma-width 64 \
  --output /campaign/my-gemm-overlap

# For serial tasks or fusion use --schedule serial-tasks or --schedule fused
# and a separate output directory. --schedule nonoverlap remains an alias for fused.

python3 scripts/pytorch_to_esp.py \
  examples/esp-validation/reusable-gemm/operators/polybench_models.py \
  --model-options '{"kernel":"twomm","dataset":"LARGE"}' \
  --mode network --compute-lanes 64 --plm-bytes 327680 --dma-width 64 \
  --output /campaign/my-reusable-2mm
```

`--network` remains an alias for `--mode network`. Omitting mode preserves the
historical network-driver default. `--schedule` is kernel-only. Network flags
also expose `--conv-lanes`, `--add-lanes` and family device-instance indices;
the compiler chooses the established GEMM default when `--compute-lanes` is
omitted. Kernel cost parameters are `--dma-request-cycles` and
`--dma-word-cycles`. These are analytical inputs, never RTL measurements.

## Compiler responsibilities

`sodap-tosa-esp-outline` extracts kernel or operator hardware.
`sodap-whole-kernel-backend` owns kernel selection, task scheduling, transport,
compute allocation, banking and LLVM emission. All three schedules use this pipeline.
Its options include schedule, lanes, PLM/reuse budgets, DMA width, analytical
request/word costs, transform library and report directory. Shared C++ helpers
also implement the matching portions of `sodap-operator-pipeline` and
`sodap-reusable-operator-backend`.

`sodap-select-kernel-tiles` keeps the historical five storage-cap proposals and
max-tile-10 proposal. Overlap evaluates the double-copy proposal set.
The fused schedule also defaults to the double-copy proposals and evaluates them on
the actual serial schedule. Set the pass or backend pipeline option
`single-copy-proposals=true` to include the experimental single-copy set. The objective is lexicographic DMA cycles,
compute iterations, then logical storage. Actual banked local storage is an
additional admission check. Historical proposals win exact ties.
`tile_search.json` identifies each candidate's proposal set, status, costs,
storage and selected tiles, plus the historical overlap-scored selection.

The serial-tasks schedule uses the overlap winner and the same separate compute
and DMA tasks. It allocates one shared slot per tile operand and checks the
actual allocation against the storage budget. The store grants work to one
receiver or compute task at a time and waits for completion. Tile publication,
reuse scopes and compute lane allocation follow the overlap path. In designs
with a packed DMA broker, the final row of each output tile returns a completion
token to the store. The comparison includes the time spent on these handoffs.

The banking pass now re-infers `memref.expand_shape` layouts after allocation
padding changes the source view. This repairs a unit-axis DOITGEN verifier
error without changing any of the 40 frozen base designs. It admits a new
DOITGEN candidate under the union search; no special DOITGEN workaround exists.

The outline/host and CPU-order passes emit and consume argument metadata.
`sodap-whole-kernel-host` retains `sodap-launches-to-esp` for kernel mode.
`sodap-tosa-network-pipeline` retains the single network host pass and
`ReusableOperatorRuntime.cpp` for all reusable operators. No host, descriptor,
adapter, socket or Bambu semantics changed in this follow-up.

## Phase 2: options and HLS contract

`sodap-apply-lcs-transform` configures the transform module in C++, replacing
shell `sed` edits. Registered C++ pipelines hold the pass sequences formerly
assembled by shell. The shell receives explicit arguments; it rejects obsolete
`LCS_*` behavior variables. The PyTorch driver isolates its options from inherited
experiment environments. `LCS_DUMP_IR` and tool-path variables remain supported.

`sodap-emit-hls-configuration` writes the unchanged `compute_config.json` and
the complete Bambu FP-resource argument to `bambu_fp_constraints.txt`. The HLS
script reads that file verbatim. Old artifacts without it must use their
archived HLS script. Python/shell still import PyTorch, export reference fixtures,
invoke tools, read compiler manifests, link, hash and report. The legacy
`select_tiles.py` and `argument_order.py` are not invoked by either clean mode.

All 40 kernel LLVM/XML/host sets and all six Exp2 plus three Exp3 host/engine
sets remain byte-identical across Phase 2. Intermediate compiler IR and stable
cost/storage reports also match. Logs and provenance records naturally differ;
the generated `sed` transform file is eliminated and the HLS-constraint file is
new. Forty fresh PyTorch lowerings reproduce the archived-TOSA results, and
the corresponding native numerical checks pass.

## Qualification limit

In the earlier union-search campaign, all overlap and MINI designs were identical to the release. LARGE nonoverlap
2MM/TRMM also remain identical. GEMM, 3MM, SYRK, SYR2K, SYMM, ATAX, GESUMMV and
DOITGEN have changed LARGE hardware and would require new qualification/builds.
Every LARGE analytical cost is no worse than the release. However, measured
GEMM MEDIUM takes 5,085,030 cycles versus 4,507,884 with release-policy tiles.
Fewer DMA beats are offset by more controller/local-memory service cycles.
No selection was changed using this measurement.

The parity audit confirms shared compute, banking, reuse and transport policies.
The schedule-dependent differences are serial task fusion, ping-pong ownership
and decoupled outstanding-load generation. Their HLS scheduling consequences
mean that the remaining gap cannot be described solely as hidden DMA latency.
