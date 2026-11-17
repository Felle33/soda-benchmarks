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

`scripts/pytorch_to_esp.py` (optionally spelled `--network`) loads the model, invokes torch-mlir, saves
test fixtures and runs tools. Python does not recognize layers, assign buffers,
emit the network schedule, optimize a datapath or generate accelerator C++.
The shell driver reads the compiler's engine manifest and invokes the common
operator backend. Packaging scripts only embed data, link objects and namespace
RTL/interface names.

The new C++ network host pass handles mixed CPU/accelerator operations and
residual lifetimes, which the original GEMM-only host pass did not support.
It is registered in the compiler and invoked by the main flow. All operator-mode inputs, including Experiment 2 GEMM chains, use this pass
and `esp_op_*`. Only whole-kernel mode retains `sodap-launches-to-esp`.
Hardware scheduling is unchanged.

## Reuse contracts and mapping

| Family | Runtime parameters | Fixed properties |
|---|---|---|
| GEMM | M/N/K, tensor addresses, row strides, source-expression descriptor fields | FP32, local tiles, banks, MAC budget |
| Dense Conv2D | N/IH/IW/IC/OC, KH/KW, OH/OW, spatial strides, padding, dilation, tensor addresses, optional ReLU | FP32, NHWC activations/OHWI weights, local tiles and resources |
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
and inputs remain persistent; reshapes repack when padding changes layout.
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
SqueezeNet1.1 retains 26 calls and byte-identical host LLVM. Device manifests
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

`OutlineConvolutions.cpp` normalizes source convolution to a structured
contraction over a **logical window view**, followed by bias/activation.
Only requested local tiles are materialized. The DMA lowering computes:

```text
input_y = output_y * stride_h + kernel_y * dilation_h - pad_top
input_x = output_x * stride_w + kernel_x * dilation_w - pad_left
```

Out-of-image coordinates become zeros without DMA transactions. OHWI weights
are read contiguously along IC and arranged in the local banked view. There is
no global im2col allocation and no convolution dispatched to the GEMM engine.
The shared compute-budget pass normalizes the local contraction with unit
kernel-reduction dimensions for MAC register blocking. This reuses a compiler
transformation, not a separate GEMM offload or handwritten LCS datapath.

`CopyToDMA.cpp` now validates a descriptor schema instead of assuming GEMM's
header. Tensor views bind dimensions to descriptor fields; convolution window
and weight views are read-only. Conv/add use 32 eight-byte slots; GEMM retains
its layout. Reserved v3 fields carry extra shape/activation bindings. Pure
matmul uses the clean source-semantics flag so absent C is never read.

The shared DMA broker retains transaction ownership and deadlock prevention.
The native ESP socket and validated thin adapter are unchanged. FP32 min/max
are legalized to integer bit comparisons/selects, including NaN and signed zero,
to avoid unsupported Bambu intrinsics. **No new Bambu patch was introduced**;
the reviewed Experiment 2 toolchain and its existing patches are prerequisites.

## Resources and limits

Defaults are conv32 MAC pairs, GEMM64 pairs and add4 lanes, with a 327680-byte
array budget per implementation. These are bounded compiler choices, not a
network performance optimizer. Convolution uses eight output pixels, 64 output
channels and up to 64 input channels, with 1x1 local kernel reductions. The IC
tile can shrink under a smaller storage budget. Addition uses a power-of-two
row tile up to 4096. GEMM retains clean Experiment 2's selection.

Reachable HLS RTL contains conv32/32, GEMM64/64 and add0/4 FP32 multiply/add
units, including shared epilogues. Local array allocations are respectively
43910, 262340 and 98691 bytes. These are not routed DSP/BRAM counts. No dynamic
integer divider module was found in this configuration's reachable RTL.

Supported source graphs use positive static FP32 tensors of rank 0..4 (rank zero for scalar coefficients);
batch-one GEMM; dense Conv2D; equal-shape add; ReLU; transpose; reshape; binary
concat; max pooling; unpadded average pooling. Unsupported operations fail
compilation. Grouped/depthwise convolution, arbitrary broadcast addition,
training, quantization and clean-path LeakyReLU are not advertised. Legacy
runtime APIs do not establish support in this new hardware path.

Operator dimensions are programmable but the host graph has compiled shapes.
Partial convolution tiles and local-buffer initialization can still do work
at fixed tile bounds; small channel counts may underutilize hardware. No old
Experiment 3 speedup is attributed to this candidate.

## Validation

The parent project's `experiments/exp-unified-20261117/README.md` holds new
unified-host qualification evidence. The read-only
`experiments/exp3-clean-20261116/README.md` holds original
commands and evidence locations. Completed checks include:

- 108 compiler regressions: descriptor schemas, extraction, explicit instance
  bindings, rejection diagnostics, FP min/max semantics and existing tests.
- Fresh ResNet18/34/SqueezeNet1.1 imports and compiler-host comparisons to
  independent PyTorch outputs. Native runtime arithmetic is an oracle, not RTL.
- Add RTL: odd rows, 4101-column tail, changing descriptors and activation.
  Conv RTL: changing stride/padding/dilation, odd channels and IC65 crossing
  the local tile. Both test DMA stalls; conv passes Icarus and Verilator.
- Actual compiler-host/three-engine RTL execution, repeated twice with stalls.
- Numerical full-system 3x3 ESP RTL using the fresh RISC-V host, CPU tensor
  operations and five sequential hardware offloads after classifier fusion.

All models produce identical convolution LLVM; the ResNets also share identical
add/GEMM LLVM. GEMM remains byte-identical to qualified clean Experiment 2.
FPGA timing/resources and new network measurements are a separate gate.
