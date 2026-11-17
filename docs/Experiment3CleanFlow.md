# Experiment 3: source-derived reusable network engines

This candidate extends clean Experiment 2 (`268d150`) to mixed Conv2D/GEMM/add
graphs. It does not use the retired Python graph compiler or hardware generators
that started from an empty MLIR module. Historical releases remain unchanged.

## Compilation responsibilities

```text
PyTorch + checkpoint -> torch-mlir/TOSA
  -> C++ layout normalization and constant affine/BatchNorm folding
  -> C++ network host lowering: SSA lifetimes, parameters, device bindings
       +-> host MLIR -> LLVM -> RISC-V object -> generic board harness
       +-> source operators -> reusable operator normalization
            -> common tiling / task extraction / LCS / ping-pong ownership
            -> descriptor DMA / tails / compute budget / banking
            -> accelerator LLVM + architecture.xml -> Bambu -> native ESP
```

`scripts/pytorch_to_esp.py --mode network` (alias `--network`) loads the model, invokes torch-mlir, saves
test fixtures and runs tools. Python does not recognize layers, assign buffers,
emit the network schedule, optimize a datapath or generate accelerator C++.
The shell driver reads the compiler's engine manifest and invokes the common
operator backend. Packaging scripts only embed data, link objects and namespace
RTL/interface names.

The common shell path takes explicit driver options and invokes registered C++
pipelines for the network host, hardware extraction and operator backend.
Transform configuration and Bambu FP constraints are compiler outputs. The
Phase 2 cleanup preserves all three network hosts and engine artifacts;
[UnifiedWholeKernelFlow.md](UnifiedWholeKernelFlow.md) records its scope and
the separate whole-kernel performance limitation.

The new C++ network host pass handles mixed CPU/accelerator operations and
residual lifetimes, which the original GEMM-only host pass did not support.
It is registered in the compiler and invoked by the main flow. All operator-mode inputs, including Experiment 2 GEMM chains, use this pass
and `esp_op_*`. Only whole-kernel mode retains `sodap-launches-to-esp`.
Hardware scheduling is unchanged.

## Reuse contracts and mapping

| Family | Runtime parameters | Fixed properties |
|---|---|---|
| GEMM | M/N/K, tensor addresses, row strides, source-expression descriptor fields | FP32, local tiles, banks, MAC budget |
| Dense Conv2D | N/IH/IW/IC/OC, KH/KW, OH/OW, spatial strides, padding, dilation, tensor addresses, optional ReLU | FP32, NHWC activations/matrix-packed persistent weights, local tiles and resources |
| Equal-shape addition | addresses, row/column counts, packed row stride, optional ReLU | FP32, local tile and addition lanes |

Bottom/right padding determine output shape; top/left padding also enter input
address calculation. Convolution memory layout is contiguous with beat-aligned
rows; arbitrary memory strides are not supported. Operator support is explicit
semantic lowering, not automatic inference of unknown operations.

`sodap.implementation` identifies compatible hardware; `sodap.instance` selects
a physical device within that family. Defaults give one implementation and
device zero per family. Multiple instances may share one implementation;
different implementations may have different compute budgets on distinct
devices. Incompatible resource assignments and conflicting device bindings are
rejected. The runtime supports synchronous rebinding.

One conv, one GEMM and one add tile is therefore the experiment configuration,
not a compiler-wide restriction. Automatic placement, replication, concurrent
launches and system-level policy optimization are outside this change.

## Host generation and weights

`Tosa/NetworkHost.cpp` consumes the actual TOSA SSA graph. It verifies supported
types/geometry, fuses single-consumer conv/add ReLU, externalizes FP32 constants
to `parameters.bin`, and allocates packed DRAM buffers using last uses and a
best-fit free list. Residual values survive their final consumer. Parameters
remain immutable; model inputs are refreshed in private arena storage each run.
Reshapes preserving element count and last dimension alias storage; row-layout
changes repack. Alias-aware last uses protect live values. Input-only CPU work
is hoisted before accelerator calls without changing the network timing region.
It emits `forward_init`, `forward_run`, `forward_close` and `network.json`.
The manifest also identifies model input buffers, parameter bindings, explicit
copies and GEMM coefficient sources, so external baseline hosts can consume
compiler decisions without reconstructing the graph.

The host reuses `Gemm`/`matchGemm` from `Tosa/Gemm.h` for full source-expression
GEMM. A batch-one classifier matmul plus bias becomes one GEMM invocation with
alpha=beta=1. Because hardware writes C in place, the host copies the persistent
bias into the writable result before **every** launch. That recurring copy is
inside the network timing interval. Neither parameters nor values with later
consumers are overwritten. Unsupported broadcasting/batches are not newly
advertised; supported non-fused cases retain separate launches.

ResNet18/34 consequently have 29/53 accelerator calls instead of 30/54.
SqueezeNet1.1 retains 26 calls; the host copy optimization changes its host LLVM
while retaining byte-identical accelerator LLVM. Device manifests
contain only families actually used; GEMM-only apps support a single-device
runtime build without unused convolution/addition devices.

Weights are copied to DRAM during initialization; their values do not specialize
hardware. `FoldConvAffine.cpp` folds constant channelwise affine chains covering
inference BatchNorm exported by torch-mlir. This reassociates FP32 arithmetic;
validation uses a numerical tolerance, not bitwise equality. It is not training
BatchNorm. Layout normalization also moves common transposes through concat.

The generic board main only initializes, calls the compiled run function,
times, validates and closes. `stage_network_app.py` copies the compiled host
object and embeds opaque data; it cannot emit layer calls. Measurement builds
disable per-operation UART logging. Timing excludes setup and validation but
includes recurring copies, CPU fallback operations and accelerator launches.

## Common hardware lowering

`OutlineAdditions.cpp` normalizes real source additions to a dynamic canonical
operation and optional activation epilogue. The common pipeline extracts and
schedules load/compute/store tasks. A pointwise register-block pass clones the
scalar region across banked lanes and guards partial lanes.

`OutlineConvolutions.cpp` represents convolution as
`out[B,P,OC] = window[B,P,R] * weights[R,OC]`, followed by bias/activation,
where `P=OH*OW` and `R=KH*KW*IC`. Batch remains a separate dimension.
This is a logical view: no im2col allocation or GEMM-engine offload occurs.
The common contraction, tile-selection, register-block and banking passes
produce the datapath; the outline specifies no fixed implementation tile.

The window DMA lowering uses row/column/channel counters for stride, padding,
dilation and row wrapping. Out-of-image elements become local zeros without a
DMA request. With dilation one it merges contiguous kernel-row segments when
the packed NHWC layout allows this. Odd IC leaves a physical padding lane per
pixel, so that gap terminates a run. A local addressing pass option can disable
run merging for verification; neither mode adds a line buffer.

Persistent OHWI weights are packed once during `forward_init`: an ordinary
strided memref view presents `[KH,KW,IC,OC]` to the existing packing runtime.
A parameter also used by a non-convolution operation retains a separate binding.
The host currently rejects non-persistent convolution weights. No weight
reorder occurs inside the recurring inference interval.

Convolution descriptor version 4 retains the geometry and adds P/R at fields
23/24, with `[R,OC]` weight rows. The descriptor has 32 eight-byte slots.
Addition and GEMM layouts are unchanged; legacy convolution runtime entry
points remain available for their old contract. The production network host
uses `esp_op_conv_matrix`. The shared descriptor reader validates the version,
so an old host cannot silently address the new weight layout.

Bounded contractions preserve semantic axis tags through common tiling and
scheduling. Compute loops, initialization, epilogues and transfers use actual
M/N/K tail extents while allocation sizes and bank factors stay static.
This opt-in mechanism currently applies to convolution. Experiment 2's
64-lane GEMM and Experiment 1 lowering retain their exact LLVM hashes.

The shared DMA broker retains transaction ownership and deadlock prevention.
The native ESP socket and validated thin adapter are unchanged. FP32 min/max
are legalized to integer bit comparisons/selects, including NaN and signed zero,
to avoid unsupported Bambu intrinsics. **No new Bambu patch was introduced**;
the reviewed Experiment 2 toolchain and its existing patches are prerequisites.

## Resources and limits

Network defaults are conv64 MAC pairs, GEMM16 pairs and add4 lanes, with a
327680-byte array budget per implementation. A GEMM-only model still defaults
to 64 lanes. Explicit driver options override these defaults; the C++ host
planner selects the default from the operator families, not Python.

Convolution uses the same candidate set, operand-reuse score, reduction tie-break
and bank factors as GEMM. At this budget both select 128x64x64; convolution
prefixes a unit batch tile and uses an 8x8 register block. Addition's selection
is unchanged to retain its protected hardware identity.

Reachable HLS RTL contains conv64/64, GEMM16/16 and add0/4 FP32 multiply/add
units (80 multipliers total). Local arrays are respectively 197510, 262340 and
98691 bytes; these exclude HLS replication and BRAM rounding. All meet their
individual budget. No dynamic integer divider is reachable. Routed DSP/BRAM
counts and achieved clock timing are separate qualification results.

Supported source graphs use positive static FP32 tensors of rank 0..4 (rank zero for scalar coefficients);
batch-one GEMM; dense Conv2D; equal-shape add; ReLU; transpose; reshape; binary
concat; max pooling; unpadded average pooling. Unsupported operations fail
compilation. Grouped/depthwise convolution, arbitrary broadcast addition,
training, quantization and clean-path LeakyReLU are not advertised. Legacy
runtime APIs do not establish support in this new hardware path.

Operator dimensions are programmable but the host graph has compiled shapes.
Convolution tails use actual bounds, but partial register blocks still leave
some arithmetic lanes idle. HLS scheduling, window address generation and local
memory work can limit utilisation. Analytical traffic reduction alone does not
establish board performance; no previous speedup is attributed to this candidate.

The separate cleanup fix skips absent device bindings in `esp_op_close`.
SqueezeNet uses convolution alone; its old inference validated but closing an
unbound add/GEMM device caused a misaligned MMIO store. This fix is independent
of the convolution datapath and was checked on the existing bitstream.

## Validation

The new campaign is `experiments/exp3-conv-tiling-20261117`. Its per-gate JSON,
raw logs and report distinguish native arithmetic oracles, HLS/component RTL,
full-system simulation and board measurements. See
[the convolution mechanism](ReusableConvolutionTiling.md) for replay commands.

All 124 inherited lit tests plus four new tests pass. New tests cover the
flattened outline and selection, window addressing/merging on and off, bounded
compute with poisoned tails, and persistent weights shared with other users.
All three networks pass both frozen inputs twice with parameters preserved.
Launch counts remain 29/53/26, with recurring repack counts 1/1/0. All eight
Experiment 2 cases pass with identical host/hardware LLVM and no timed repacks;
all ten Experiment 1 MINI lowerings are unchanged.

The numerical Verilator layer tests report 29,560,893 cycles for the first
ResNet layer and 18,955,943 for a 56x56 64-to-64 3x3 layer, against required
baselines of 47,980,000 and 35,950,000 cycles. These no-stall component results
are not FPGA measurements. Current full-system and board qualification status
is recorded in the campaign evidence, not inferred from native tests.
