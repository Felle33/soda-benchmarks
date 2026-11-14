# Experiment 1 — final measured release, 14 November 2026

This is the final paper-data release for Experiment 1, promoted after all 20
LARGE FPGA cases passed full-output numerical validation. It is an immutable
reference for these measurements; future compiler work should use another
branch and output directory. Experiment 2 and 3 remain separate evaluations.

Compiler: [WilliamBaisi/soda-benchmarks, release/esp-exp1-final-20261114](https://github.com/WilliamBaisi/soda-benchmarks/tree/release/esp-exp1-final-20261114).
Tagged artifacts: [WilliamBaisi/bambu-esp-example, exp1-final-20261114](https://github.com/WilliamBaisi/bambu-esp-example/releases/tag/exp1-final-20261114)
(private repository; Tommaso already has access).

## What is final

- Ten PolyBench kernels, LARGE, FP32, overlap and nonoverlap LCS: 20 designs.
- Native 2x2 ESP on VC707-03: CPU/accelerator 50 MHz, NoC 100 MHz, DMA 64 bits.
- 64 core FP32 multiply/add pairs per accelerator, with separately counted
  coefficient arithmetic. Core compute is matched; total area/BRAM is not.
- 40 HLS builds (MINI/LARGE), 20 MINI full-system RTL tests, 20 LARGE component
  checks, and all 20 timing-closed bitstreams with full-output FPGA validation.
- Primary time is accelerator launch/wait including DMA and polling. Host
  allocation/packing/unpacking and end-to-end times are separate CSV columns.
- Two GEMM calibration runs per mode, then one run per other case under the
  agreed sampling policy. These are local SPECTRA-derived baseline measurements,
  not the original paper's numbers. Baseline numerical/bitstream evidence is included.

The overlap designs beat SPECTRA in 9/10 kernels, geometric mean 9.48x including
3MM; individual winning speedups are 3.90x–36.37x. 3MM remains 1.73x slower.
Overlap improves on nonoverlap by 2.09x–12.24x (geometric mean 4.37x), using
1.60x–4.22x SPECTRA's BRAM, 42–53% fewer LUTs, and similar/slightly fewer DSPs.
This preserves the motivation to evaluate reusable GEMM offloads for 3MM.

## Find the evidence

| Path | Contents |
|---|---|
| `data/experiment1-results.csv` | Performance, numerical coverage, arithmetic units, LUT/FF/DSP/BRAM; all 20 cases |
| `data/spectra-resource-comparison.csv` | Same-scope accelerator resource comparison |
| `data/completion-summary.json` | Verified recorded evidence and timing scope |
| `artifacts/campaign/candidate-v2/board/<mode>-v2/<kernel>/` | Raw UART logs and per-run results |
| `artifacts/campaign/candidate-v2/fpga/<kernel>/<mode>/` | Exact measured bitstream, routed checkpoint and reports |
| `artifacts/campaign/candidate-v2/generated/<MINI or LARGE>/<kernel>/` | Inputs, references, MLIR/LLVM, architecture XML, Bambu Verilog and memory initialization files |
| `artifacts/campaign/candidate-v2/apps/` | Generated host, compiled ELF/binary, link commands and runtime adaptation |
| `artifacts/spectra/` | Baseline bitstreams, reports, host binaries and raw board results |
| `compiler.bundle` | Complete compiler Git history for offline recovery |
| `bambu/patches/series` | Seven reviewable patches against Bambu cda4db2 |
| `bambu/before`, `bambu/after` | Exact original/final source contexts; `verification.json` checks the series |
| `flow/` | Exact campaign scripts, checker fixes, regression logs and implementation record |
| `FILES.json` | Per-file SHA256, size, and explicit symlink targets |

Original records retain their historical absolute paths. `release.json` maps
them to this package. The original compiler snapshot is intentionally unchanged:
its old candidate-status field describes when it was captured; this release's
promotion and final FPGA evidence supersede that status without altering hashes.

## Run on emilia as Tommaso

```bash
cd /home/tommaso/esp-exp1-final-20261114
python3 release/verify.py --compiler compiler
bash release/emilia.sh tests
bash release/emilia.sh replay gemm
bash release/emilia.sh synthesize gemm
bash release/emilia.sh lower gemm
```

`tests` runs the compiler/graph tests. `replay` recompiles the preserved MINI
Verilog with the packed native-DMA adapter and checks four invocations per mode,
including a 50,000-cycle write stall. `synthesize` runs the final Bambu driver on
the preserved LLVM for both modes, then those component RTL tests. `lower`
regenerates both schedules from the saved TOSA. These smoke commands do not
program the FPGA. New outputs go to `runs/`, never to the measured artifacts.
Use `bash release/emilia.sh build` to rebuild the plugin from source.

The pinned compiler image and Tommaso's `/home/tommaso/opt/bambu` installation
must exist on emilia. The runner overlays our isolated driver/plugins; it does
not modify the installation. The full-system and FPGA campaign scripts in
`flow/` record the original ESP platform/CAD paths; their commands require the
installed Questa/Vivado/RISC-V environment and the original platform fixture.
This release includes the original full-system records and final routed
artifacts, while component replay is independently relocatable.

## Download or recover

Download all assets from the tagged release. Verify `SHA256SUMS` before
extracting archives. If an archive is split into `.part-000`, `.part-001`, etc.,
concatenate the parts in filename order before `zstd -d` / `tar -x`.
Extract artifact archives inside this release directory. The compiler can be
restored with `git clone compiler.bundle ../compiler`. Copy
`artifacts/compiler-build/plugin` into `../compiler/build/plugin` for the exact
prebuilt plugin, or rebuild it in the pinned environment.

The recorded base Docker image is available by repository digest; the exact
local image and environment metadata are recorded in `environment.json`.
Licensed CAD tools are provided by emilia and are not distributed in this release.

## Changes to review

Preserves Tommaso's lowering and our DMA admission/write-readiness, bounded
reuse, reusable operators and thin adapter. Adds shared compute-budgeted
two-level blocking/banking and packed transport: two FP32 values per 64-bit
beat, aligned layouts, safe boundary read/modify/write and ownership, host
packing and reusable descriptor updates. Compiler layouts cover 64/128/256/512;
only 64-bit physical Bambu/FPGA is qualified here. ESP socket RTL is unchanged.

Bambu changes: stale partition/interface handling, transformed architecture
selection, equivalent-index residue handling, bank-pointer normalization,
switch-address hoisting, same-block interface access ordering, and defined
invalid nonblocking FIFO returns/state completion. All seven patches apply to
the captured base and reproduce the recorded final sources. They remain local
patches proposed for Tommaso/Fabrizio review, not upstream fixes.
