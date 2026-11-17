#!/usr/bin/env bash
# Tool orchestration for registered kernel and operator pipelines.
# Usage: tosa_to_lcs.sh input.mlir output [--mode kernel|network] [options]
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
input=$(realpath "${1:?input TOSA}")
mkdir -p "${2:?output directory}"
out=$(realpath "$2")
original_options=("${@:3}")
source "$repo/scripts/lcs/options.sh" "${original_options[@]}"
if [[ $mode == network && -z $family ]]; then
  exec bash "$repo/scripts/lcs/network.sh" "$input" "$out" "${original_options[@]}"
fi
if [[ $mode == kernel && -n $family ]]; then
  echo 'A hardware family is only valid for network engine extraction' >&2; exit 1
fi
source "$repo/scripts/check_docker.sh"
cd "$out"
plugin_path=${MLIR_PLUGIN_PATH:-$repo/build/plugin/lib/SODAPlugin.so}
plugin=(--load-dialect-plugin="$plugin_path" --load-pass-plugin="$plugin_path")
tosa_plugin=(--load-pass-plugin="${SODAP_TOSA_PLUGIN:-$(dirname "$plugin_path")/SODATosaPlugin.so}")
step() {
  local log=$1; shift
  local flags=(--mlir-disable-threading)
  [[ ${LCS_DUMP_IR:-false} == true ]] && flags+=(--mlir-print-ir-after-all --mlir-print-ir-module-scope)
  $DOCKER_RUN "$@" "${flags[@]}" 2>> "$log" || { tail -40 "$log" >&2; return 1; }
}
for file in frontend acc host cpu; do : > "ir_after_each_pass_$file.mlir"; done
: > ir_after_each_pass.mlir
outstanding=true; sequential=false; overlap=true
[[ $schedule == nonoverlap ]] && { outstanding=false; sequential=true; overlap=false; }
[[ -n $family ]] && outstanding=false
printf 'dma_word_stride=2\nprofile=%s\noutstanding_loads=%s\nsequential=%s\ndma_abi=packed\ndma_data_width=%s\n' \
  "$profile" "$outstanding" "$sequential" "$dma_width" > layout.txt
[[ -n $family ]] && printf 'tiles=operators\n' >> layout.txt
printf 'reuse_buffer_budget=%s\nplm_budget=%s\noverlap=%s\n' "$reuse_bytes" "$plm_bytes" "$overlap" >> layout.txt
outline_family=${family:-kernel}
step ir_after_each_pass_frontend.mlir mlir-opt "$input" "${tosa_plugin[@]}" \
  --pass-pipeline="builtin.module(sodap-tosa-esp-outline{family=$outline_family lanes=$lanes plm-bytes=$plm_bytes})" -o outlined.mlir
if [[ -z $family ]]; then
  step ir_after_each_pass_acc.mlir mlir-opt outlined.mlir "${plugin[@]}" \
    --pass-pipeline="builtin.module(sodap-whole-kernel-backend{schedule=$schedule lanes=$lanes dma-width=$dma_width plm-bytes=$plm_bytes reuse-bytes=$reuse_bytes request-cycles=$request_cycles word-cycles=$word_cycles transform-file=$transform_file report-dir=$out})" -o acc.mlir
else
  step ir_after_each_pass_acc.mlir mlir-opt outlined.mlir "${plugin[@]}" \
    --pass-pipeline="builtin.module(sodap-operator-pipeline{family=$family lanes=$lanes dma-width=$dma_width plm-bytes=$plm_bytes reuse-bytes=$reuse_bytes transform-file=$transform_file directory=$out})" -o acc.mlir
fi
for dir in tiles/*/; do
  $DOCKER_RUN llvm-link -S - "$dir/ac_channel_specializations.ll" < "$dir/acc.ll" \
    | $DOCKER_RUN opt -passes=always-inline -S -o "$dir/05_hw_acc.ll"
done
cp tiles/forward_kernel/05_hw_acc.ll tiles/forward_kernel/architecture.xml .
# Reporting only; the compiler has already emitted the selected sizes/costs.
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);costs=[p/'schedule_cost.json'] if (p/'schedule_cost.json').exists() else sorted(p.glob('tiles/*/schedule_cost.json'))
with (p/'tile_cost.txt').open('w') as out:
 for f in costs:
  c=json.loads(f.read_text());prefix='' if f.parent==p else f.parent.name+': '
  out.write(prefix+'dma_cycles = %d, plm_bytes = %d\n'%(c['dma_cycles'],c['plm_bytes']))
PY
if [[ -n $family ]]; then
  python3 "$repo/scripts/lcs/native_dma_contract.py" record "$out" --hardware-only
  exit 0
fi
bash "$repo/scripts/lcs/check_host_mode.sh" "$mode"
step ir_after_each_pass_host.mlir mlir-opt acc.mlir "${plugin[@]}" \
  --pass-pipeline="builtin.module(sodap-whole-kernel-host{profile=$profile dma-width=$dma_width directory=$out})" -o host.mlir
host=host.mlir
if [[ $profile == true ]]; then
  step ir_after_each_pass_host.mlir soda-opt host.mlir "${plugin[@]}" --pass-pipeline='builtin.module(sodap-profile-kernel)' -o host_profiled.mlir
  host=host_profiled.mlir
fi
step ir_after_each_pass_host.mlir soda-opt "$host" --lower-all-to-llvm=use-bare-ptr-memref-call-conv \
  --convert-func-to-llvm --reconcile-unrealized-casts -o host_llvm.mlir
$DOCKER_RUN mlir-translate --mlir-to-llvmir host_llvm.mlir -o 05_kernel_riscv.ll
mkdir -p cpu
step ir_after_each_pass_cpu.mlir mlir-opt "$input" "${plugin[@]}" --pass-pipeline='builtin.module(sodap-kernel-cpu-frontend)' -o cpu/bufferized.mlir
step ir_after_each_pass_cpu.mlir soda-opt cpu/bufferized.mlir "${plugin[@]}" --convert-all-to-soda -soda-outline-bambu-code -o cpu/outlined_raw.mlir
step ir_after_each_pass_cpu.mlir soda-opt cpu/outlined_raw.mlir "${plugin[@]}" \
  --pass-pipeline="builtin.module(sodap-order-kernel-arguments{argument-order-file=$out/cpu/argument_order.txt})" -o cpu/outlined.mlir
step ir_after_each_pass_cpu.mlir soda-opt cpu/outlined.mlir "${plugin[@]}" \
  -soda-extract-arguments-to-c-testbench=using-bare-ptr -soda-generate-bambu-accelcode=no-aa -o cpu/kernel.mlir
cmp argument_order.txt cpu/argument_order.txt
cpu=cpu/kernel.mlir
if [[ $profile == true ]]; then
  step ir_after_each_pass_cpu.mlir soda-opt "$cpu" "${plugin[@]}" --pass-pipeline='builtin.module(sodap-profile-kernel)' -o cpu/kernel_profiled.mlir
  cpu=cpu/kernel_profiled.mlir
fi
step ir_after_each_pass_cpu.mlir soda-opt "$cpu" --lower-all-to-llvm=use-bare-ptr-memref-call-conv \
  --convert-func-to-llvm --reconcile-unrealized-casts -o cpu/kernel_llvm.mlir
$DOCKER_RUN mlir-translate --mlir-to-llvmir cpu/kernel_llvm.mlir -o 05_cpu_baseline.ll
python3 "$repo/scripts/lcs/native_dma_contract.py" record "$out"
cat ir_after_each_pass_{frontend,acc,host,cpu}.mlir > ir_after_each_pass.mlir
