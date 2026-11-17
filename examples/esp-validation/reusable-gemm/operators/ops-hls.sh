#!/usr/bin/env bash
# In the container: Bambu HLS of <generated dir>, copied to <hardware dir> (must not exist).
set -euo pipefail
gen=$1; hw=$2
test ! -e "$hw"; cp -r "$gen" "$hw"
export BAMBU_DEVICE=xc7vx485t-2ffg1761 BAMBU_CLOCK_PERIOD=20
start=$(date +%s)
bash /workspace/scripts/ll_to_verilog_esp_version.sh "$hw/05_hw_acc.ll" forward_kernel > "$hw/../hls.log" 2>&1 && rc=0 || rc=$?
echo "HLS rc=$rc seconds=$(( $(date +%s) - start ))"
exit $rc
