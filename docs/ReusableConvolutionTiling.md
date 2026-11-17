# Reusable convolution through the common contraction flow

This candidate starts at `a04bdcf`, preserving the unified operator host and
its reshape aliases, GEMM C ownership and input-only scheduling. Convolution
continues to use a distinct ESP device. It becomes a matrix-shaped operation
inside MLIR so the existing resource policy can select its implementation.

## Compiler mechanisms

1. `Tosa/OutlineConvolutions.cpp` extracts runtime geometry and builds a batched
   contraction `[B,P,R] x [R,OC] -> [B,P,OC]`. P spans image rows; R spans kernel
   rows, columns and input channels. Bias and ReLU remain the epilogue. The
   reduction order stays KH/KW/IC, and batch remains supported.
2. `SelectReusableHardware.cpp` applies the existing GEMM candidate search and
   score to P/OC/R. It prefixes batch tile 1, without adding a convolution policy.
   At 327680 bytes the choice is 128/64/64. The banked allocation report remains
   the final admission check. Addition selection is deliberately unchanged.
3. `PackedViews.inc` maps window coordinates directly to packed NHWC addresses.
   Tile-entry coordinates are decoded by counter loops, then advanced with
   carries. Invalid image coordinates produce local zero values. Dilation-one
   contiguous segments merge up to a row, tile, image or physical packing gap.
   No dynamic divider, line buffer or materialized im2col tensor is introduced.
4. `sodap.bound_axes` and `sodap.bound_axis` preserve semantic axes through
   tiling, scheduling and prefix splitting. `RuntimeTileBounds.h` derives
   `min(capacity, upper-origin)` from these loops. Common register blocking,
   initialization and epilogues consume these extents; backing banks remain
   static. Outside-tail zero-fill is omitted only for opted-in bounded
   contractions. In-image-window padding still emits zero. The protected GEMM
   and whole-kernel pipelines do not opt in and remain byte-identical.
5. `NetworkHost.cpp` packs persistent weights once at setup using a generic
   strided view and the existing copy-in API. A weight with an ordinary tensor
   consumer has two bindings so neither interpretation corrupts the other.
   Dynamic/non-persistent weights are rejected. The recurring host graph retains
   its previous launch counts and copies.
6. Convolution descriptor v4 adds P/R in slots 23/24 and uses `[R,OC]` weights.
   `esp_op_conv_matrix` emits this contract. GEMM/add descriptors and the ESP
   native-DMA adapter/socket are unchanged. No Bambu patch is added.

Defaults are conv64, network GEMM16 and add4. GEMM-only models retain GEMM64.
Explicit driver arguments remain available; default selection is in C++.
These resource defaults and the tile heuristic are not a network mapping search.

## Qualification and reproduction

Run from the parent campaign (`experiments/exp3-conv-tiling-20261117`). The
container pins the toolchain and mounts previous campaigns read-only. Set
`LCS_DUMP_IR=false` for network compilations. Use new output directories.

```bash
bash container.sh bash -c 'cmake --build build/plugin -j4'
bash container.sh bash -c 'python3 -c "from lit.main import main; main()" -sv build/plugin/test'
bash container.sh python3 scripts/pytorch_to_esp.py \
  examples/esp-validation/networks/smoke.py --model-options '{"case":"residual"}' \
  --output /campaign/replay-residual
bash container.sh env BAMBU_DEVICE=xc7vx485t-2ffg1761 BAMBU_CLOCK_PERIOD=12.8 \
  bash scripts/ll_to_verilog_esp_version.sh \
  /campaign/replay-residual/generated/engines/conv_default/05_hw_acc.ll forward_kernel
bash container.sh python3 examples/esp-validation/networks/conv_capture.py \
  /campaign/replay-capture --suite component
bash container.sh python3 examples/esp-validation/networks/conv_rtl.py \
  /campaign/replay-residual/generated/engines/conv_default \
  /campaign/replay-capture /campaign/replay-rtl
```

Use `--simulator icarus` for the second simulator. Capture suites `first-layer`
and `body-layer`, with RTL `--stalls 0`, measure the required full-size layers.
The testbench passively observes arithmetic and DMA handshakes; it does not
modify the generated RTL or adapter. Its four cycle buckets are DMA only,
floating-point work in flight only, both, and neither. The last bucket includes
local-memory accesses, address computation, synchronization and control; it is
not evidence that all those cycles could be eliminated.

The shared HLS arithmetic audit checks requested units, physical local arrays
and reachable dynamic dividers. The complete task also requires unchanged
Exp1/Exp2 artifacts, two-input repeated native checks, full-system 3x3 RTL,
routed timing/capacity and board numerical validation. Component speedups must
not be described as network board speedups. See the campaign's `evidence/`
and `REPORT.md` for completed and outstanding gates.

The independently approved cleanup fix (`86ee404`) skips unbound devices in
`esp_op_close`. It was tested using the old FPGA image before the new
convolution datapath; SqueezeNet's previously valid inference now exits cleanly.
