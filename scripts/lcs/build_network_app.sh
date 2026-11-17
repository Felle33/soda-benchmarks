#!/usr/bin/env bash
# Link the compiler-produced host object with a network-independent board main.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
input=$(realpath "${1:?compiled network directory}")
app=${2:?new app directory}
esp=$(realpath "${3:?ESP root}")
soc=$(realpath "${4:?ESP SoC directory}")
stage_args=()
if [[ -n ${NN_SECOND_FIXTURE:-} ]]; then stage_args+=(--second-fixture "$NN_SECOND_FIXTURE"); fi
python3 "$repo/scripts/lcs/stage_network_app.py" "$input" "$app" "${stage_args[@]}"
app=$(realpath "$app")
env NN_MAIN_C="$repo/examples/esp-validation/networks/baremetal_main.c" \
  NN_SEPARATE_DEVICES=1 NN_QUIET=${NN_QUIET:-1} NN_DRAM_BYTES="${5:-16777216}" \
  bash "$repo/examples/esp-validation/reusable-nn/build-software.sh" "$app" "$esp" "$soc"
python3 - "$app" <<'PY'
import hashlib,json,sys
from pathlib import Path
app=Path(sys.argv[1]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(app/'build.json').write_text(json.dumps(dict(status='LINKED_NOT_BOARD_VALIDATED',
    host_source='C++ MLIR host lowering; generic C harness',
    files={p.name:sha(p) for p in app.iterdir() if p.is_file()}),indent=2)+'\n')
PY
