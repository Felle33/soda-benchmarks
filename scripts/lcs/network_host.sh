#!/usr/bin/env bash
# Tool orchestration only: graph semantics, layouts, storage and calls are MLIR
# passes. This development entry validates the host; it does not qualify RTL.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
input=$(realpath "${1:?input TOSA}")
mkdir -p "${2:?output directory}"
out=$(realpath "$2")
source "$repo/scripts/lcs/options.sh" "${@:3}"
plugin=${SODAP_LIB_DIR:-$repo/build/plugin/lib}/SODATosaPlugin.so
mlir-opt "$input" --load-pass-plugin="$plugin" \
  --pass-pipeline="builtin.module(sodap-tosa-network-pipeline{directory=$out dma-width=$dma_width conv-instance=$conv_instance add-instance=$add_instance gemm-instance=$gemm_instance conv-lanes=$conv_lanes gemm-lanes=$gemm_lanes add-lanes=$add_lanes plm-bytes=$plm_bytes})" \
  -o "$out/host.mlir"
soda-opt "$out/host.mlir" --lower-all-to-llvm=use-bare-ptr-memref-call-conv \
  --convert-func-to-llvm --reconcile-unrealized-casts -o "$out/host-llvm.mlir"
mlir-translate --mlir-to-llvmir "$out/host-llvm.mlir" -o "$out/host.ll"
"${CLANG:-clang-19}" -O3 -ffp-contract=off -fPIC -shared "$out/host.ll" \
  "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp" \
  -I"$repo/examples/soda-plugins/include" -o "$out/native.so"
"${RISCV_CLANG:-clang-19}" --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d \
  -mcmodel=medany -O2 -c "$out/host.ll" -o "$out/host-riscv.o"
