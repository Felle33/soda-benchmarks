#!/usr/bin/env bash
# Explicit tool options. Defaults mirror registered compiler pipeline defaults.
# This parser does not select tiles, banks, blocks, operators or schedules.
mode=kernel
schedule=overlap
family=
lanes=64
plm_bytes=327680
reuse_bytes=262144
dma_width=64
request_cycles=64
word_cycles=1
profile=true
conv_lanes=64
gemm_lanes=0
add_lanes=4
conv_instance=0
gemm_instance=0
add_instance=0
transform_file="$repo/scripts/lcs/transform.mlir"
for name in ${!LCS_@} ${!_LCS_@}; do
  [[ $name == LCS_DUMP_IR ]] && continue
  echo "$name is no longer a compiler option; use explicit driver/pass options" >&2
  exit 1
done
while (($#)); do
  [[ $# -ge 2 ]] || { echo "Missing option value: $1" >&2; exit 1; }
  case "$1" in
    --mode) mode=$2 ;; --schedule) schedule=$2 ;; --hardware-family) family=$2 ;;
    --compute-lanes) lanes=$2 ;; --plm-bytes) plm_bytes=$2 ;; --reuse-bytes) reuse_bytes=$2 ;;
    --dma-width) dma_width=$2 ;; --request-cycles) request_cycles=$2 ;; --word-cycles) word_cycles=$2 ;;
    --profile) profile=$2 ;; --transform-file) transform_file=$2 ;;
    --conv-lanes) conv_lanes=$2 ;; --gemm-lanes) gemm_lanes=$2 ;; --add-lanes) add_lanes=$2 ;;
    --conv-instance) conv_instance=$2 ;; --gemm-instance) gemm_instance=$2 ;; --add-instance) add_instance=$2 ;;
    *) echo "Unknown compiler tool option: $1" >&2; exit 1 ;;
  esac
  shift 2
done
[[ $mode == kernel || $mode == network ]] || { echo 'mode must be kernel or network' >&2; exit 1; }
[[ $schedule == overlap || $schedule == nonoverlap ]] || { echo 'schedule must be overlap or nonoverlap' >&2; exit 1; }
[[ $profile == true || $profile == false ]] || { echo 'profile must be true or false' >&2; exit 1; }
[[ $dma_width == 64 ]] || { echo 'Bambu backend supports DMA64; wider layouts require direct MLIR tests' >&2; exit 1; }
[[ -z $family || $family == gemm || $family == conv || $family == add ]] || { echo 'Unsupported hardware family' >&2; exit 1; }
for value in "$lanes" "$plm_bytes" "$reuse_bytes" "$request_cycles" "$word_cycles" "$conv_lanes" "$gemm_lanes" "$add_lanes" "$conv_instance" "$gemm_instance" "$add_instance"; do
  [[ $value =~ ^[0-9]+$ ]] || { echo 'Compiler budgets/instances must be unsigned integers' >&2; exit 1; }
done
