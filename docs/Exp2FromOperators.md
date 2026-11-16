# Experiment 2 from the model's operators

2026-11-16, branch `exp2-operators-packed`.

Experiment 2's reusable GEMM accelerator is now generated from the model's
TOSA operators by the generic dataflow passes, not by a dedicated generator.
It runs the same launches as William's engine, with the same tile, lanes and
storage. In component RTL it is faster in every case measured (MINI 3mm:
80,469 cycles against 86,496). LARGE board results, routing and full-system
RTL are not measured yet.

## What changed

All changes are on branch `exp2-operators-packed` of
`esp-exp2-exp3-final-20261115/exp2/compiler`, started from
`release/esp-exp2-final-20261114` (William's measured release, compiler
`4d2d932`).

| Commit | Part | Change |
|---|---|---|
| `78643aa` | Patch | Packed DMA in operators mode (the patch, without group tokens) |
| `b0bea1e` | G1 | Operators mode takes the tile from the hardware budget (`reusable_gemm_config.py`): 128×64×64 for 64 lanes and 327,680 B |
| `b0bea1e` | G2 | Run-time tails: `tile_and_fuse` keeps static tiles that need not divide the shapes; new `sodap-dataflow-tails` cuts each transfer to `min(tile, extent − offset)` and zero-fills a local tile only when partial; the DMA lowering requests the cut rows |
| `8527c23` | G3 | `sodap-tosa-outline-operators` outlines each GEMM (matmul, optional alpha mul, optional add of C or beta·C) as one launch |
| `8527c23` | G4 | `sodap-join-multiple-tiles` builds one tile `(A, B, C, M, K, N, alpha, beta) -> D` in TOSA; new `sodap-linalg-fold-runtime-broadcasts`; `@tasks_gemm` puts matmul and epilogue in one task |
| `8527c23` | G5 | C is read only when beta ≠ 0 (new `sodap-dataflow-guard-reads`); the body selects alpha·AB when beta = 0; f32 descriptor fields |
| `8527c23` | G6 | Host launches pass the real alpha and beta and copy C into D's place in the arena |
| `8527c23` | G7 | `scripts/lcs/check_gemm_oracles.py`: launches, one hardware hash, storage |
| `a7dbe01` | Epilogue | The epilogue runs only on the valid part of each tile; the dataflow passes treat a part of a static tile as an access of the tile |
| `ab36adc` | Tools | Lowering, HLS and RTL drivers and the RTL timeline probe (`examples/esp-validation/reusable-gemm/operators/`) |

The frozen release, its runs and its logs were not modified; every new run is
in a new directory under `exp2/runs/`.

## The new pipeline

```
William's flow                              This branch (LCS_TILES=operators)
--------------                              ---------------------------------
model TOSA                                  model TOSA
  |  tosa-to-linalg, ReusableGemmToESP        |  sodap-tosa-outline-operators   [GEMM pattern: matmul,
  |  (pattern match on linalg)                |                                  alpha mul, beta*C add]
  v                                           |  sodap-join-multiple-tiles       [one tile, shapes as args,
host: esp_gemm_* calls                        |                                  TOSA body with select(beta==0)]
                                              v
empty module                                tile: tosa-to-linalg
  |  sodap-generate-reusable-gemm             |  sodap-linalg-fold-runtime-broadcasts   [new]
  |  (C++ writes the GEMM loops,              |  linalg-fuse-elementwise-ops
  |   bounds min(tile, extent - offset))      |  transform @tasks_gemm: create_tasks, tile_and_fuse
  v                                           |    [static 128x64x64 tiles; epilogue on the valid part]
dataflow passes                               v
  load-store, schedule, ping-pong,          dataflow passes
  copy-to-dma (+ DMA broker)                  load-store, schedule, ping-pong   [parts of static tiles]
  v                                           report-cost
nodes -> LLVM -> Bambu                        sodap-dataflow-guard-reads        [new: C only if beta != 0]
                                              sodap-dataflow-tails              [new: cut transfers]
                                              copy-to-dma (packed, no broker)
                                              v
                                            nodes -> compute budget, banking, register block -> LLVM -> Bambu
                                            host: sodap-launches-to-esp -> esp_gemm_* calls (real alpha/beta, C copy-in)
```

The tile sizes come from the hardware budget only (D1), so all six lowerings
(MINI and LARGE gemm, 2mm, 3mm) give one `05_hw_acc.ll`; shapes, addresses,
strides, alpha and beta reach the accelerator through the descriptor at each
launch.

## Validation

Everything up to component RTL passes on both sides; full-system RTL, routing
and the board are William's only so far.

| Check | William's release | Ours |
|---|---|---|
| Compiler tests | 94 lit | 101 lit + 15 Python |
| One hardware LLVM for the six lowerings | Yes | Yes (`cd4b60e3…`) |
| Launches equal to the release's `invocations.json` (M/N/K, offsets, alpha, beta) | Reference | Yes on all six; MINI captures byte-identical |
| Storage within 327,680 B | 262,392 B | 262,340 B |
| Bambu HLS: 64 FP mul + 64 FP add, II = 1 | Yes | Yes |
| Component RTL, MINI gemm/2mm/3mm, stall 0 and 50,000 | Yes | Yes |
| Odd-tail, multi-tile, 300,000-cycle stall | Yes | Yes |
| beta = 0 with C all NaN | Not tested | Yes; C's reads skipped (51 against 71 read requests) |
| Full-system ESP SoC RTL | Yes | Started |
| Vivado routing, resources, timing | Yes | Started |
| LARGE on the VC707 board | Yes | No |

Re-running William's archived RTL in our harness reproduces his recorded
cycles exactly (stress case: 276,638 / 382,881 / 458,141).

## Numbers

In the packed DMA64 component testbench our engine is faster at stall 0 in
every case and level with William's at stall 50,000. Cycles are cumulative
over a workload's launches.

| Case | William's engine | Ours, first version | Ours, now |
|---|---:|---:|---:|
| MINI gemm, stall 0 | 41,189 | 75,265 | **29,023** |
| MINI 2mm, stall 0 | 28,179 / 66,217 | 73,858 / 147,820 | **26,338 / 53,358** |
| MINI 3mm, stall 0 | 28,011 / 57,845 / 86,496 | 73,753 / 148,138 / 221,992 | **26,233 / 53,750 / 80,469** |
| MINI gemm, stall 50,000 | 51,930 | 75,265 | 50,989 |
| MINI 2mm, stall 50,000 | 51,125 / 102,582 | 73,858 / 147,820 | 50,569 / 101,305 |
| MINI 3mm, stall 50,000 | 51,125 / 102,638 / 153,983 | 73,753 / 148,138 / 221,992 | 50,569 / 101,333 / 152,013 |
| Stress (129×65×67, 129×17×65, 31×66×9), stall 0 | 276,638 / 382,881 / 458,141 | 339,827 / 514,798 / 656,576 | **200,707 / 290,532 / 346,152** |
| Stress, 300,000-cycle write stall | 454,363 / 788,293 / 1,111,597 | 479,186 / 783,717 / 1,087,538 | 383,604 / 688,136 / 991,959 |

"First version" is the engine before the epilogue fix (`8527c23`); "now" is
`a7dbe01`.

The hardware, side by side (Bambu HLS, xc7vx485t-2ffg1761, 20 ns):

| | William's engine | Ours |
|---|---|---|
| Tile, lanes | 128×64×64, 8×8 block, 64 lanes | Same |
| FP units in the compute node | 64 mul + 64 add | Same |
| Padded local storage | 262,392 B | 262,340 B |
| Task nodes | 4: load, compute, store, DMA broker | 3: load, compute, store |
| FIFOs between nodes | 21 | 6 |
| Verilog | 65,220 lines | 56,297 lines |
| C when beta = 0 | Not read | Not read |
| Epilogue | Valid elements only | Valid elements only |
| Load order | C, then A and B | A and B, then C during the matmul |

## Why the numbers differ

Measured with a timeline probe added to the testbench (FIFO handshakes
between the task nodes, node start/done, DMA requests), on both engines. The
compute task's time per phase, in cycles:

| Phase | MINI 3mm, first launch: William / ours first / ours now | Stress, first launch: William / ours now |
|---|---|---|
| Waiting for the A/B tiles (descriptor, zero fill, DMA) | 14,968 / 13,768 / 13,768 | 144,664 / 60,479 |
| Matmul | 10,135 / 10,131 / 10,131 | 81,080 / 89,369 |
| Epilogue | 1,765 / 49,283 / 1,763 | 50,846 / 50,838 |

- **The first version's 2.6×:** its epilogue ran on the whole static 128×64
  tile, 8,192 elements at about 6 cycles each (49,283 cycles); William's runs
  only on `min(M − i, 128) × min(N − j, 64)` (1,765). Load and matmul were
  already equal. Fixed in `a7dbe01`.
- **MINI gemm (beta ≠ 0):** William's engine loads C before A and B, so the
  matmul waits for C too (A/B ready at cycle 26,060); ours loads C during the
  matmul (A/B ready at 14,858, C at 23,787, matmul done at 24,989).
- **Multi-tile (stress):** our ping-pong overlaps loads with compute, so the
  compute task waits far less for A/B. Our matmul is slower when loads run
  beside it: about 11k cycles a K step against 10.1k. The likely cause, not
  verified: the two slots of a ping-pong buffer are one array, whose ports
  load and compute share.
- **Stall 50,000:** both engines are bound by the stall; the totals are within
  2%.

## Not measured yet

- **LARGE performance:** William's numbers are board seconds (gemm 2.21 s,
  2mm 2.96 s, 3mm 4.81 s at 50 MHz); we have no LARGE RTL (tens of millions of
  cycles per case) and no board run.
- **Area and timing closure:** William's engine routes at 41,010 LUT, 14,960
  registers, 142 DSP, 235 BRAM36 (WNS +0.555 ns). Ours has one node and 15
  FIFOs fewer, but has not been through Vivado.
- **Full-system ESP RTL:** William ran the six MINI cases in the SoC; ours are
  starting next to his runs.

## Where things are

| What | Where |
|---|---|
| Compiler changes | branch `exp2-operators-packed`, commits above |
| Pipeline plan and status | `docs/Exp2PipelinePlan.md` |
| Six lowerings, HLS, RTL, oracles (current engine) | `exp2/runs/epi-six-2/` |
| First-version engine and its runs | `exp2/runs/g3-six-2/` |
| Probed RTL runs of both engines | `exp2/runs/perf-1/` |
| beta = 0 / NaN-C case | `exp2/runs/g8-nan-1/` |
| IR after each of the 150 passes (MINI gemm) | `exp2/runs/print-after-all-1/cases/MINI/gemm/generated/passes/` |
| Drivers and probe | `examples/esp-validation/reusable-gemm/operators/` |
| William's release | `releases/exp2/` |
