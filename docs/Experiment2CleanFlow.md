# Experiment 2: reusable operators through the unified host compiler

Operator mode has one host generator: `sodap-tosa-network-host` in
`Tosa/NetworkHost.cpp`, linked with `ReusableOperatorRuntime.cpp` (`esp_op_*`).
GEMM chains and mixed CNN graphs use the same host lowering and runtime.
Whole-kernel mode (`--mode kernel`, Experiment 1) retains
`sodap-launches-to-esp`; operator-mode drivers reject that legacy host route.

## Compile a PyTorch program

The factory returns `(torch.nn.Module, inputs)`. Inputs are contiguous FP32
tensors, including scalar coefficient tensors. For example:

```bash
python3 scripts/pytorch_to_esp.py \
  examples/esp-validation/reusable-gemm/operators/polybench_models.py \
  --model-options '{"kernel":"twomm","dataset":"LARGE"}' \
  --mode network \
  --compute-lanes 64 --plm-bytes 327680 --dma-width 64 \
  --output /campaign/cases/LARGE/twomm
```

The frontend executes the actual model for reference data and imports fresh
TOSA. Canonical 2MM retains its alpha/beta expressions; `twomm_mm` is not used.
Python exports fixtures and invokes tools. It makes no fusion, scheduling,
tiling, banking or allocation decision.

The shared shell path now takes explicit options and invokes registered C++
pipelines (`sodap-tosa-network-pipeline`, `sodap-tosa-esp-outline`,
`sodap-operator-pipeline`). Transform configuration and complete Bambu FP
constraints are emitted by compiler passes, not shell edits or Python policy.
Only IR dumping and tool locations remain environmental. See
[UnifiedWholeKernelFlow.md](UnifiedWholeKernelFlow.md) for the cleanup and
identity evidence; this migration leaves Exp2 hardware and hosts unchanged.

Outputs under `generated/` include `host.mlir`, `host.ll`, `host-riscv.o`,
`parameters.bin`, `network.json`, and `engines/<implementation>/05_hw_acc.ll`
and `architecture.xml`. `--network` is a compatibility spelling, not a second
host path. `--network-host-only` omits hardware lowering.

## Shared source-expression recognition and runtime

`include/sodap/Tosa/Gemm.h` contains the existing `Gemm`/`matchGemm` matcher,
shared by hardware outlining and host planning. It recognizes the imported
source expression `alpha * (A @ B) + beta * C`, respecting single-use boundaries.
The host schedules the launch after its inputs exist and uses SSA last uses to
keep intermediate products in DRAM. GEMM, canonical 2MM and 3MM use 1, 2 and 3
GEMM launches, respectively, on the same engine.

`esp_op_gemm_full_source` accepts coefficients and explicit C presence. Scalar
model inputs are read on each run; constants remain compiler-extracted data.
The GEMM engine writes its result in place into its C buffer. The host consumes
C's arena buffer when its last read is this invocation and its packed row layout
matches the result. Model inputs are copied into the arena on every run, so their
private copies can be consumed while the caller's inputs remain unchanged.
Persistent parameters, later readers and C that aliases another GEMM operand
require a distinct writable result. Alias chains extend the storage lifetime.
`esp_op_gemm_source` remains the pure-product entry point.

Reshapes preserving element count and last dimension are storage aliases;
reshapes changing the packed row layout retain a repack. A stable topological
partition moves CPU operations depending only on inputs/parameters before the
first accelerator launch, including protected C preparation when independent of
accelerator results. Allocation and lifetime analysis use this execution order.
These are host transformations; they do not change operator hardware or fusion.

Both entry points use the existing packed source-semantics GEMM descriptor v3.
Field 14 contains the FP32 C-presence flag, independent of beta. Explicit `0*C`
still reads C, preserving NaN propagation; an absent C is not read. The 128-byte
GEMM descriptor is unchanged. The shared operator arena reserves a 256-byte
header, so data addresses differ from the old GEMM-only arena, while descriptor
address units and field meanings remain identical.

`network.json` is the compiler's actual ordered execution plan: buffers,
input/parameter bindings, copies, coefficient sources, dimensions, C presence,
engine implementations and device instances. The manual SPECTRA baseline
serializes this manifest into tables consumed by a generic C interpreter; it
does not reconstruct GEMM chains by benchmark name. Its existing padded FP32
transport differs from the generated engine's packed FP32 transport.

GEMM-only manifests contain only the GEMM family. The operator runtime supports
both separate typed sockets and a build configured for the existing single
GEMM device. It does not require convolution or addition devices for these apps.

## Hardware pipeline remains unchanged

The host pass preserves the source module for the existing hardware extraction:
`TosaTransposes.cpp`, `JoinMultipleTiles.cpp`, `SelectReusableHardware.cpp`,
tiling, task scheduling, ping-pong ownership, guarded/tail DMA, compute-budget
selection, banking and the registered reusable-operator backend. The matcher
was factored into a shared header without changing hardware outlining behavior.

64 compute lanes and a 327680-byte array budget retain the existing GEMM tile
selection. No Bambu patch, adapter, socket, descriptor reader or FPGA design
change is part of host unification. Existing accelerator images may be reused
only after byte-identical accelerator LLVM and architecture checks pass.

## Measurement and qualification

The primary Experiment 2 interval is first GEMM invocation entry through final
completion, including dispatch, DMA, polling and all work between invocations.
Input packing, C copies before the first invocation, final unpacking, setup,
UART and numerical validation are outside this interval and must be reported
separately. Copies between launches are inside it. The whole `forward_run`
function is therefore not the Experiment 2 timing boundary.

`SODAP_OP_PROFILE` enables separate allocation, packing and unpacking counters.
Allocation covers context/page-table setup, tensor-layout metadata and teardown;
packing covers parameter/input copies and required repacks; unpacking covers
the final output copy. These accumulated categories can overlap the primary
interval if a required repack occurs between launches. They do not redefine it.

Qualification requires original and added compiler tests; MINI/LARGE native
GEMM/2MM/3MM; odd dimensions; explicit NaN-C; descriptor byte comparisons;
unchanged whole-kernel outputs; and MINI full-system ESP RTL with the new host
and the existing GEMM RTL. FPGA results must identify the new host binary and
must not inherit measurements from the old `esp_gemm_*` host path.

FP32 validation uses stated tolerances, not bitwise equality with PyTorch.
Shapes are static at import; dimensions, addresses and coefficients are runtime
accelerator parameters. True batched GEMM and general broadcast addition retain
existing unsupported-case diagnostics. See `Experiment3CleanFlow.md` for the
shared mixed-network capabilities and classifier fusion.
