#!/usr/bin/env bash
# Build the generated host against three independent generated RTL engines.
set -euo pipefail
here=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd -- "$here/../../.." && pwd)
network=$(realpath "${1:?compiled network}")
hardware=$(realpath "${2:?directory containing conv/add/gemm packaged subdirectories}")
mkdir -p "${3:?output directory}"
out=$(realpath "$3")
args=(-fPIC -std=c++17 -O2 -pthread -DSODAP_OP_SEPARATE_RTL -DSODAP_OP_EXTERNAL_EXECUTOR
      -I"$repo/examples/soda-plugins/include" -I/usr/share/verilator/include -I/usr/share/verilator/include/vltstd)
libraries=()
for kind in conv add gemm; do
  package=$hardware/$kind/packaged
  top=cnn_${kind}_bambu_basic_dma64
  verilator --cc --build -Wno-fatal -j 4 -CFLAGS -fPIC --prefix V$kind \
    --Mdir "$out/$kind" --top-module "$top" "$package/engine.v" "$package/$top.sv" \
    > "$out/build-$kind.log" 2>&1
  args+=(-I"$out/$kind")
  libraries+=("$out/$kind/V${kind}__ALL.a")
  cp "$package"/*.mem "$out/"
done
"${CLANG:-clang-19}" -O2 -fPIC -c "$network/generated/host.ll" -o "$out/host.o"
c++ "${args[@]}" -shared "$out/host.o" \
  "$repo/examples/soda-plugins/lib/sodap/ExecutionEngine/ReusableOperatorRuntime.cpp" \
  "$here/../reusable-nn/rtl_executor.cpp" "${libraries[@]}" \
  /usr/share/verilator/include/verilated.cpp /usr/share/verilator/include/verilated_threads.cpp \
  -o "$out/host-rtl.so"
sha256sum "$network/generated/host.ll" "$out/host-rtl.so" "$out"/*.mem > "$out/inputs.sha256"
