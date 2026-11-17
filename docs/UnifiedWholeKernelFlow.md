# Unified kernel/network entry point (local, unqualified candidate)

The local `experiment/flow-unified-20261117` branch starts at `ba29c43`.
The whole-kernel migration is implemented, but qualification stopped at gate I:
LARGE nonoverlap scheduled DMA cost regresses for 2MM, TRMM and DOITGEN.
No new HLS, FPGA implementation or board campaign was started. Phase 2 is pending.
The campaign report is `../REPORT.md` relative to this checkout's root; machine
readable gate evidence is `../evidence/`.

Use `scripts/pytorch_to_esp.py --mode kernel --schedule overlap|nonoverlap`
for a whole-kernel accelerator, and `--mode network` (alias `--network`) for
reusable operators. Both accept a PyTorch factory, model options and a fresh
output directory. The existing PolyBench factory now accepts all ten kernels.
The kernel mode retains `sodap-launches-to-esp`; network mode retains
`sodap-tosa-network-host` and `ReusableOperatorRuntime.cpp`.

`sodap-whole-kernel-backend` is the registered kernel pipeline. Its options
include `schedule`, `lanes`, `plm-bytes`, `reuse-bytes`, `dma-width`,
`request-cycles`, `word-cycles`, `transform-file`, and `report-dir`.
`sodap-select-kernel-tiles` evaluates the original five storage-cap proposals
and historical max-tile-10 proposal through the existing passes. It ranks
scheduled DMA cost, then compute iterations, then logical storage. Physical
banked storage is checked independently. The serial proposal model counts one
copy of transported/yielded tiles; overlap counts two. It records the old
opposite/overlap-scored reference as well, without using measured/HLS results.

`tile_search.json` records candidates, failures, costs, tiles and banked
storage; `tile_sizes.txt` is emitted from transform metadata. Argument-order
files are emitted by the outline/host and CPU-order passes, rather than Python
IR parsing. Shared compute/banking/LLVM helpers preserve operator hardware and
host identity. Tests cover both selection schedules, all-candidate rejection,
outline ABI metadata and argument-order files.

All 20 overlap designs and 10 MINI serial designs reproduce the frozen LLVM,
XML and host. Ten LARGE serial LLVM files change due to the tile-copy model;
all XML and host files remain identical. All 132 lit tests pass. Exp2 six
host/engine cases and Exp3 three network host/engine sets remain identical.

The stop has two causes. The finite proposal set can omit the older better
choice after changing the copy model (2MM/TRMM). DOITGEN's five storage-based
proposals expose an existing unit-axis `memref.expand_shape` layout-verifier
failure in the compute/banking path, leaving the expensive max-tile-10 fallback.
These issues have not been tuned around or fixed beyond the specified plan.

Remaining Phase 2 work is behavior-selecting environment plumbing, operator
transform sed editing, and compiler emission of Bambu FP constraints. The old
Python helpers remain for explicit legacy reproduction but are not invoked by
either new mode. Do not treat this candidate as qualified or promote its costs
into experimental results.
