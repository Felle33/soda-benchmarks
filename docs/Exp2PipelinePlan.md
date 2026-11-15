# Experiment 2: reusable GEMM offload for 3MM

2026-11-15. One accelerator generated from the model's matmul, invoked three
times. Compared with SPECTRA and with Experiment 1's per-kernel 3MM, which is
still 1.73× slower than SPECTRA. Built on `LCS_TILES=operators`.

## Decisions (2026-11-16, branch `exp2-operators-packed`)

| # | Decision | Consequence |
|---|---|---|
| D1 | Tile chosen as in the last Experiment 2 run: from the hardware budget only, `reusable_gemm_config.py` with `LCS_COMPUTE_LANES=64`, `LCS_PLM_BYTES=327680` → 128×64×64 | Same tile for every workload; run-time tails are required. This replaces the divisors-of-100 choice below (and step 3's 100×100×100) once operators mode has tails; until then the sizes must still divide every invocation |
| D2 | C loaded as in the last Experiment 2 run: the C tile is read only when `beta ≠ 0` | Hardware guard on beta |
| D3 | The ping-pong schedule is Experiment 1's, unchanged: the variant that loads A and B before publishing tokens is not applied on this branch and is not planned | Operators mode keeps Experiment 1's dataflow schedule; cycle counts measured with that variant are not reported here |

## What exists today

| Piece | State | Where |
|---|---|---|
| Operators mode for `tosa.matmul` | Done for tile sizes dividing every invocation; threemm SMALL passes RTL | `sodap-tosa-outline-operators`, `sodap-join-multiple-tiles`, `sodap-launches-to-esp` |
| Packed DMA64 (2 FP32 per beat) | In fork PR #3, not merged; review asks for changes | `CopyToDMA`, `DmaTransport.h`, `ESPPasses.cpp` |
| 64-lane compute budget, banking, register block | In fork PR #3; matmul only | `ComputeBudget.cpp`, `BankedMemrefs.cpp` |
| FPGA campaign scripts of Experiment 1 | Available, outside this repo | `esp-exp1-final-20261114/release/flow/`, `docs/ESPExperiment1Final.md` |
| Bambu with the review series (patches 0001–0008) | Built, in the release's `sdk/` | `build-pinned-bambu.py`, `bambu-source-manifest.json` |
| torch-mlir | In the pinned image | `/opt/torch-mlir` (torch 2.4.0.dev20240604, as the fixtures) |

## Step 0: land the prerequisites

1. **Merge the parts of PR #3 that this experiment uses:** packed DMA
   (68657d6) and the compute budget (45e4d5e, e02f876, 428c9b5). The
   broker is untested: avoid it with even tile sizes along each transfer's
   row (see below). Aligned column offsets and lengths need no
   read-modify-write, so CopyToDMA adds no broker (`CopyToDMA.cpp:428`).
2. **Use the Experiment 1 FPGA scripts**
   (`esp-exp1-final-20261114/release/flow/`), so this experiment uses the same
   Vivado/board runner and measurement as Experiment 1 (accelerator
   launch/wait including DMA; host copies in separate columns).

## Pipeline

`01_tosa.mlir` → `tosa_to_lcs.sh` with `LCS_TILES=operators
LCS_DMA_ABI=packed LCS_COMPUTE_LANES=64` → Bambu (`sdk/`, 20 ns,
`xc7vx485t-2ffg1761`) → component RTL → MINI full system → LARGE bitstream →
VC707.

## Why the tile sizes are divisors of 100

One accelerator runs all three matmuls. Its tile sizes are fixed in hardware,
and operators mode has no partial tiles. So each tile size must divide that
dimension in **every** invocation. 3MM LARGE
(`benches/PolyBenchPyTorch/linear_algebra/utils.py`: ni 800, nj 900, nk 1000,
nl 1100, nm 1200):

| Invocation | m | n | k |
|---|---:|---:|---:|
| E = A·B (800×1000 · 1000×900) | 800 | 900 | 1000 |
| F = C·D (900×1200 · 1200×1100) | 900 | 1100 | 1200 |
| G = E·F (800×900 · 900×1100) | 800 | 1100 | 900 |
| **Greatest common divisor** | **100** | **100** | **100** |

Each of tile-m, tile-n and tile-k is therefore a divisor of 100: 1, 2, 4, 5,
10, 20, 25, 50 or 100. The three can differ.

Two further limits:

- **Packed DMA:** the transferred row length is k for A and n for B and C.
  Tile-k and tile-n must be even so every transfer covers whole 64-bit beats,
  which avoids read-modify-write and the broker. Tile-m may be odd.
- **The 8×8 register block always has a tail.** No divisor of 100 is a
  multiple of 8, so the block's last rows or columns are partly idle and the
  banked buffers are padded (100 → 104, 50 → 56). At 100, utilization is
  100/104 ≈ 96% per dimension. A 4×16 block makes tile-m tail-free (4 divides
  100) but leaves n at 100/112. Pick by the HLS result, and check the PLM
  after padding (`compute_storage.json`).

## Steps

| # | Step | Work | Done when |
|---|---|---|---|
| 1 | Packed DMA in operators mode | Relax `scripts/tosa_to_lcs.sh:151` (stride 2 required) when `LCS_DMA_ABI=packed`. The packed descriptor path is already in CopyToDMA (fields in bytes, `GEMM_PACKED_VERSION`), and so is the host path (`esp_gemm_open_packed`, padded `ld`) | threemm SMALL passes the packed DMA64 harness (`reusable-gemm/run-adapter.sh`, `dma64_tb.sv`) with and without the 50,000-cycle stall |
| 2 | Compute budget in operators mode | None expected: the tile pipeline is shared (`tosa_to_lcs.sh:382`), so `LCS_COMPUTE_LANES` already applies. Check that `RegisterBlockMatmul` and banking accept the joined tile (static local buffers, run-time outer bounds) | Bambu reports 64 FP mul + 64 FP add in the compute node, II=1; RTL passes |
| 3 | LARGE tile sizes | Choose among the divisors of 100 above, with tile-k and tile-n even. `LCS_TILE_SIZES`, or the cost model with `LCS_PLM_BYTES` | HLS fits the PLM budget after padding |
| 4 | RTL qualification | Component RTL at MINI and LARGE (selected tiles at LARGE, as in Experiment 1), both stall settings. MINI full system on native 2×2 ESP (`reusable-gemm/check-system.py`) | 3 invocations correct, intermediate E and F stay in the arena |
| 5 | FPGA | Vivado bitstream with timing closed, board run on VC707-03 with the step 0.2 runner | Full-output check passes; launch/wait cycles recorded |
| 6 | Results | One row next to Experiment 1's 3MM rows: CPU, SPECTRA, per-kernel overlap/nonoverlap, reusable GEMM. Report LUT/FF/DSP/BRAM: one engine against three matmuls' worth of SPECTRA hardware. Optionally William's template engine at the same budget and tiles | Table and resource CSV in the campaign directory |

## Risks

- Packed DMA plus operators mode together are untested; step 1 is the first
  place this combination runs.
- The register-block tail pads local buffers and leaves some lanes idle
  (see above); step 3 checks the PLM.

## Status 2026-11-16 (branch `feature/exp2-reusable-gemm`)

| Step | State |
|---|---|
| 0.1 PR #3 | Cherry-picked 45e4d5e, e02f876, 428c9b5, 68657d6; lit 94/94. Kernel mode byte-identical on gemm and threemm SMALL (`05_hw_acc.ll`, `architecture.xml`, `05_kernel_riscv.ll`, `05_cpu_baseline.ll`); `layout.txt` gains `dma_abi`/`dma_data_width` (from PR #3) |
| 0.2 FPGA scripts | Available: `esp-exp1-final-20261114/release/flow/` |
| 1 packed + operators | Done. Script: stride check only for the legacy ABI; operators mode writes `tiles=operators` in `layout.txt`, which `run-native.sh`/`run-adapter.sh` now accept; Make stamps `LCS_DMA_ABI`, `LCS_COMPUTE_LANES`, ... when set. threemm SMALL `[[10,10,10]]`: native 3.8e-6, packed DMA64 RTL (Questa) PASS, 3 invocations (measured on `feature/exp2-reusable-gemm`, with a schedule this branch does not have, see D3; to be repeated here) |
| 2 compute budget | Compiles: 8×8 block, 64 `fmul` in node1, banked, native PASS. **Bambu (stock /opt/panda) segfaults** after interface binding: CSROA expands node arguments and Bambu reads the stale `architecture.xml`, which William's Bambu patches 0001/0002 fix (`7d84255:docs/experiment1-final/bambu-patches/`). The release's `sdk/` Bambu has that series (0001–0008); HLS against it is the next check |
| 3 LARGE tiles | **100×100×100** (compile only, 64 lanes): 288,195 B padded ≤ 327,680, lowest DMA cost. HLS check waits for step 2 |
| 4 RTL | MINI `[[2,2,2]]` (gcd 2), no lanes: native PASS, RTL PASS both stalls. LARGE RTL and the full system (needs `ESP_ROOT`, compiled SoC) open |
| 5, 6 | Open (need step 2) |

Cycles (Questa, total of the three invocations' boundary cycles; measured on
`feature/exp2-reusable-gemm`, to be repeated on this branch, see D3):

| threemm | Stall 0 | Stall 50,000 |
|---|---:|---:|
| MINI, packed DMA64, `[[2,2,2]]` | 98,816 | 246,260 |

With one FP lane the compute dominates; the gain from packed DMA should come
with step 2's 64 lanes.

LARGE sweep (64 lanes, packed; `tile_cost.txt` DMA cycles, padded bytes):

| m×n×k | Padded PLM (B) | DMA cycles |
|---|---:|---:|
| 100×100×100 | 288,195 | 149,397,600 |
| 100×100×50 | 204,995 | 166,677,600 |
| 50×100×100 | 188,995 | 220,677,600 |
| 100×50×100 | 190,595 | 239,615,200 |
| 50×50×50 | 76,195 | 362,735,200 |
| 25×100×100 | 139,395 | 363,237,600 |
| 20×20×20 | 12,995 | 1,417,468,000 |

Scripts (untracked, `my_trials/exp2/`): `build.sh` (container: lowering,
testdata, native trace, Bambu), `questa.sh` (host: the packed harness in
Questa), `sweep.sh` (LARGE tiles). MINI/LARGE TOSA are SMALL's with the
dimensions substituted (`my_trials/exp2/inputs/`); they can be regenerated
with the torch-mlir of the pinned image (`/opt/torch-mlir`).

## Order and critical path

Step 0, then steps 1 → 6. Every step reuses existing passes; the long pole
is the FPGA campaign (step 5).
