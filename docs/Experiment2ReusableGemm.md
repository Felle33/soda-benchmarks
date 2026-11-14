# Experiment 2 reusable GEMM mechanism

Experiment 2 binds canonical GEMM, 2MM and 3MM to one runtime-programmable GEMM
implementation. It branches from `exp1-final-20261114`; the final Experiment 1
measurements, hardware and compiler release remain the reference.

## Compiler behavior

`scripts/tosa_to_reusable_gemm.sh` first bufferizes and outlines the original
TOSA graph. `ReusableGemmToESP.cpp` extracts an ordered list of invocations and
one shared DRAM arena. Tensor dimensions, addresses, row strides, alpha and beta
are runtime descriptor fields. Intermediate matrices stay in that arena between
launches; the host copies initial inputs once and final outputs once.

Canonical scalar multiplication after a matmul and a following scaled residual
addition are recognized as GEMM epilogues. Their order is retained: for 2MM,
`T = alpha * (A B)` followed by `O = T D + beta * C`. The pass rejects unsupported
generic operations and uses of an intermediate that would make this fusion
change observable behavior. It does not replace canonical 2MM with two unscaled
products. `invocations.json` exports the same preflighted buffers, argument
bindings and calls used to generate the host.

Hardware generation is independent of the graph. With `LCS_COMPUTE_LANES=64`,
`GenerateReusableGemm.cpp` retains a local `linalg.matmul`. The existing
compute-budget, banking, register-blocking and dataflow passes lower it. No
handwritten C++ datapath or post-HLS Verilog edits implement the compute engine.
The campaign's simple shape-independent storage heuristic selects local tiles
128 by 64 by 64 and an 8 by 8 register block under a 327680-byte PLM budget.
The storage report is 262392 bytes before HLS replication and BRAM rounding.
These are implementation choices, independent of the extracted descriptors.

Partial input tiles are zero-filled in local storage before valid elements are
loaded. Static local contractions retain their fixed bounds; DMA and output
stores use the valid runtime extents. This permits invocation dimensions that
do not divide the local tiles. Both load/compute/store ownership and the packed
DMA request broker use the existing common lowering. The default generator
without a compute budget retains the previous scalar implementation.

## Runtime and measurement

`ReusableGemmRuntime.cpp` retains descriptor validation and per-instance
ownership. `SODAP_GEMM_PROFILE` adds accumulated allocation, packing, invocation
and unpacking timing plus an optional `esp_gemm_profile_event` observer.
`SODAP_GEMM_QUIET` suppresses progress messages during measurement. Errors remain
visible. The campaign observer records counters in memory and prints afterward.

Primary time is first invocation entry to final invocation completion, including
programming, DMA, compute, polling and host gaps. Input loading, allocation,
packing, unpacking, UART and numerical validation are outside that interval.
Single-launch durations and host gaps are also retained.

## Baseline and reproducibility

The existing SPECTRA-derived GEMM bitstream is unchanged. Its new bare-metal host
interprets the compiler invocation manifest, programs the existing registers and
launches the same engine one, two or three times. It keeps intermediate matrices
in DRAM. This is a separate C host backend, not the MLIR-generated host binary.
The original padded FP32 DMA layout is retained; the generated engine uses
packed FP32. Equal core compute does not imply equal transport, BRAM or total
area, and these differences must be reported.

The Experiment 2 campaign is under
`/home/william/bambu-esp-example/experiments/exp2-reuse-20261114` on emilia.
Its README, `STATE.json`, `campaign-state.json`, validation logs and build
manifests distinguish implemented mechanisms, passing checks and FPGA results.
One generated RTL image is tested across all MINI/LARGE descriptor shapes;
qualification requires native reference agreement, HLS resource auditing,
component RTL including tails/stalls/dependencies, and MINI full-system RTL.

The pinned Bambu build needs an additional dual-port array interface fix.
When a shared array is discovered first through its reader, write ports are
appended in a different order from the order assumed by the HDL generator.
The original generator connects write-data zero to write-enable one and vice
versa. The arithmetic result is correct but the stored data can be zero.
Patch `0008-dp-array-port-order.patch` resolves the four write outputs by their
semantic names. The campaign preserves failing and passing RTL evidence and
rebuilds the pinned Bambu source plus all eight patches. It does not modify
Tommaso's installation, the native ESP socket, or the frozen Experiment 1 tools.
