# Experiment 2: model-derived reusable operators

This replaces the November 14 dedicated-generator campaign. Base: Tommaso's
`exp2-operators-packed` at `8c3d413`. The archived results remain historical;
this implementation needs its own FPGA qualification and measurements.

## Compile a PyTorch program

The input file exports `create_model(**options) -> (torch.nn.Module, inputs)`.
Inputs are contiguous FP32 tensors, including scalar tensor coefficients. The
current backend accepts GEMM chains and one tensor result. Example, in the
pinned compiler environment:

```bash
python3 scripts/pytorch_to_esp.py \
  examples/esp-validation/reusable-gemm/operators/polybench_models.py \
  --model-options '{"kernel":"twomm","dataset":"LARGE"}' \
  --compute-lanes 64 --plm-bytes 327680 --dma-width 64 \
  --output /campaign/cases/LARGE/twomm
```

Use a fresh output directory. The frontend executes the actual PyTorch model
to generate reference values and imports it through torch-mlir. It never reads
archived TOSA, picks tiles, or substitutes a hardware implementation. The
original canonical 2MM includes its alpha/beta epilogues. `twomm_mm` is not used.

The compiler produces `generated/05_hw_acc.ll`, `architecture.xml`,
`05_kernel_riscv.ll`, `05_cpu_baseline.ll`, `invocations.json`, compute/storage
reports, and IR after each pass. The hardware LLVM is Bambu's input; the host
LLVM links with the reviewed ESP runtime and a bare-metal application harness.

## Compiler mechanisms and ownership

Tommaso's work is retained:

- `TosaTransposes.cpp` recognizes model GEMM expressions and outlines launches.
- `JoinMultipleTiles.cpp` normalizes compatible GEMMs to a dynamic-shape TOSA
  operator, binds descriptor fields, and joins launches to one implementation.
- `SODAPTransforms.cpp`, `LoadStore.cpp`, `Schedule.cpp`, `PingPong.cpp` and
  `Prefix.h` derive tiled tasks, demand-driven schedules, local storage and
  safe ownership tokens from the operator.
- `GuardReads.cpp`, `Tails.cpp` and `CopyToDMA.cpp` implement conditional reads,
  partial-tile transfers/zero fills and packed DMA. Epilogues execute on valid
  tile extents.
- `ComputeBudget.cpp` and `BankedMemrefs.cpp` allocate the contraction budget,
  infer compatible banking, expose register-blocked compute and lower to LLVM.
- `ESPPasses.cpp` emits the shared arena and ordered host launches. Intermediate
  matrix products remain in DRAM between invocations.

New consolidation:

- `SelectReusableHardware.cpp` implements the previous bounded outer-tile
  heuristic as `sodap-select-reusable-hardware`. It uses only lane and storage
  budgets; 64 lanes/327680 bytes select 128x64x64. The final banked storage
  report remains an independent admission check.
- `tile_and_fuse` consumes the compiler's `sodap.hardware_tile_sizes` attribute.
  Explicit transform sizes retain precedence for experiments outside this CLI.
- The initial Python register-block selector is removed from `tosa_to_lcs.sh`.
  The common C++ compute-budget pass selects every contraction's dimensions.
- `sodap-reusable-operator-backend` is a registered C++ pipeline covering
  guarded/tail DMA, task-to-function conversion, compute/banking and LLVM.
- `sodap-launches-to-esp` exports the same checked invocation graph it uses for
  host code. The manual baseline consumes this manifest; it does not reconstruct
  a benchmark-specific sequence.

The new entry point cannot fall back to `GenerateReusableGemm.cpp` or
`tosa_to_reusable_gemm.sh`. Neither `reusable_gemm_config.py`, `compute_budget.py`
nor `select_tiles.py` participates in the new reusable pipeline. The latter
remains a legacy whole-kernel-mode search helper; consolidation of that search
is separate from this Experiment 2 replacement. Python is used for the PyTorch
frontend, test harnesses and tool orchestration, with no dataflow optimization.
The common shell driver rejects the superseded `LCS_REUSABLE_GEMM=true` switch.
Generated C++ `ac_channel` binding units are transport ABI support; they do not
implement the numerical datapath.

## FP32 source expression and descriptor v3

The old normalization treated `beta == 0` as "do not read C". This is not
equivalent to the explicit source expression `beta*C` when C contains NaNs.
The new entry point selects `source-semantics=true` when outlining operators.
It carries an explicit f32 C-presence flag (0 or 1) through MLIR and descriptor
field 14. It skips C only when the source operator has no C expression. Explicit
`0*C` still reads C and participates in the result.

Descriptor version 3 preserves the packed v2 tensor layout and 128-byte header.
`esp_gemm_invoke_source` writes the new version/flag; compiler-generated hardware
checks that version. No ESP socket, CSR, DMA port or adapter changes are needed.
Versions 1/2 and `esp_gemm_invoke` retain their archived behavior. The low-level
TOSA pass defaults to that behavior for old tests; the supported PyTorch driver
and operators-mode shell flow default to source semantics. `LCS_GEMM_FP_SEMANTICS=blas`
is explicit compatibility mode, not the replacement experiment.

Arithmetic remains FP32. Validation uses stated tolerances against PyTorch;
this is not a claim of bitwise identity across floating-point implementations.
The NaN test checks that the newly introduced semantic distinction is preserved
through both the generated host and synthesized hardware.

## Scope and qualification

The mechanism currently normalizes GEMM with batch dimension one. Source
tensor shapes are static at tracing/lowering time; the accelerator receives
runtime M/N/K, strides, addresses and coefficients. Identical hardware for
different traced workloads does not imply general dynamic PyTorch support.
Unsupported CPU uses of intermediates or C sourced from a previous launch are
diagnosed by host lowering. Convolution/addition require separate operator
contracts; this change does not claim to add those contracts.

One generated LLVM/RTL image must cover MINI/LARGE GEMM/2MM/3MM and unseen odd
multi-tile shapes. Qualification includes native reference checks, compiler
regressions, exact HLS core-unit counting, finite/NaN/tail/stalled-DMA RTL tests,
full-system MINI validation on the matched 2x2 SoC, routing/timing checks and
full-output LARGE FPGA validation. Physical DSP and BRAM usage are measured,
not inferred from logical lane or array-byte counts.

The campaign keeps the existing pinned Bambu build and all eight reviewed
patches, including the dual-port array interface correction. No new Bambu or
handwritten datapath modifications are required by this consolidation.
