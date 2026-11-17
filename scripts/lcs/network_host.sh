#!/usr/bin/env bash
# Tool orchestration only: graph semantics, layouts, storage and calls are MLIR
# passes. This development entry validates the host; it does not qualify RTL.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
input=$(realpath "${1:?input TOSA}")
mkdir -p "${2:?output directory}"
out=$(realpath "$2")
plugin=${SODAP_LIB_DIR:-$repo/build/plugin/lib}/SODATosaPlugin.so
mlir-opt "$input" --load-pass-plugin="$plugin" \
  --pass-pipeline="builtin.module(func.func(sodap-tosa-sink-transposes,tosa-layerwise-constant-fold,sodap-tosa-sink-transposes,tosa-layerwise-constant-fold,sodap-tosa-fold-conv-affine,sodap-tosa-sink-transposes,tosa-layerwise-constant-fold,sodap-tosa-fold-conv-affine),canonicalize,sodap-tosa-network-host{directory=$out dma-width=${LCS_DMA_DATA_WIDTH:-64} conv-instance=${LCS_CONV_INSTANCE:-0} add-instance=${LCS_ADD_INSTANCE:-0} gemm-instance=${LCS_GEMM_INSTANCE:-0} conv-lanes=${LCS_CONV_LANES:-32} gemm-lanes=${LCS_GEMM_LANES:-${LCS_COMPUTE_LANES:-64}} add-lanes=${LCS_ADD_LANES:-4} plm-bytes=${LCS_PLM_BYTES:-327680}})" \
  -o "$out/host.mlir"
soda-opt "$out/host.mlir" --lower-all-to-llvm=use-bare-ptr-memref-call-conv \
  --convert-func-to-llvm --reconcile-unrealized-casts -o "$out/host-llvm.mlir"
mlir-translate --mlir-to-llvmir "$out/host-llvm.mlir" -o "$out/host.ll"
"${CLANG:-clang-19}" -O3 -ffp-contract=off -fPIC -shared "$out/host.ll" \
  "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp" \
  -I"$repo/examples/soda-plugins/include" -o "$out/native.so"
"${RISCV_CLANG:-clang-19}" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d \
  -mcmodel=medany -O2 -c "$out/host.ll" -o "$out/host-riscv.o"
