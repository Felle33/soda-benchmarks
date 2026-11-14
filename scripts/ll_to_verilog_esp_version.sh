#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input.ll> <top-fname>" >&2
  exit 1
fi

INPUT=$(realpath "$1")
DIRNAME_INPUT="$(dirname "${INPUT}")"
TOP_FNAME="$2"
BAMBU=${BAMBU_BIN:-bambu}
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
source "$SCRIPT_DIR/check_docker.sh"

mkdir -p "$DIRNAME_INPUT"/bambu_syn
cd "$DIRNAME_INPUT"/bambu_syn

cp "$INPUT" accelerator.ll
cp "$DIRNAME_INPUT/architecture.xml" architecture.xml
args=(accelerator.ll --architecture-xml=architecture.xml
  --top-fname="$TOP_FNAME" --compiler=I386_CLANG19
  --generate-interface=INFER --generate-components=separate
  --clock-period="${BAMBU_CLOCK_PERIOD:-10}" --no-clean -v3)
if [[ -n ${BAMBU_DEVICE:-} ]]; then args+=(--device="$BAMBU_DEVICE"); fi
if [[ -f "$DIRNAME_INPUT/compute_config.json" ]]; then
  fp_units=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["fp_units_per_task"] if c.get("enabled", True) else 0)' "$DIRNAME_INPUT/compute_config.json")
  [[ $fp_units =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Invalid FP resource configuration' >&2; exit 1; }
  if (( fp_units > 0 )); then
    args+=("-C=__float_mule8m23b_127nih=$fp_units,__float_adde8m23b_127nih=$fp_units")
  fi
  cp "$DIRNAME_INPUT/compute_config.json" .
fi
printf '%q ' "$BAMBU" "${args[@]}" > command.txt
printf '\n' >> command.txt
$DOCKER_RUN "$BAMBU" --version > bambu-version.txt 2>&1
sha256sum accelerator.ll architecture.xml > inputs.sha256
$DOCKER_RUN "$BAMBU" "${args[@]}" > bambu.log 2>&1 || {
  tail -40 bambu.log >&2
  exit 1
}
test -s "$TOP_FNAME.v"
# The top can depend on technology RTL and initialized memories beside it.
find . -maxdepth 1 -type f \( -name '*.v' -o -name '*.mem' \) -print | sort > rtl-files.txt
while IFS= read -r file; do sha256sum "$file"; done < rtl-files.txt > rtl.sha256
echo "Bambu RTL and dependencies: $PWD (see rtl-files.txt)"
