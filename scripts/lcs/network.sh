#!/usr/bin/env bash
# Common compiler network target. Python here only reads a compiler manifest
# and hashes artifacts; it does not recognize operators or choose a schedule.
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
input=$(realpath "${1:?input TOSA}")
mkdir -p "${2:?output directory}"
out=$(realpath "$2")
source "$repo/scripts/lcs/options.sh" "${@:3}"
bash "$repo/scripts/lcs/network_host.sh" "$input" "$out" "${@:3}"
python3 - "$out/network.json" > "$out/engines.tsv" <<'PY'
import json,sys
for engine in json.load(open(sys.argv[1]))['engines']:
    print(engine['name'],engine['operator'],engine['lanes'],engine['plm_bytes'],sep='\t')
PY
while IFS=$'\t' read -r name family lanes storage; do
  bash "$repo/scripts/tosa_to_lcs.sh" "$out/source.mlir" "$out/engines/$name" \
    --mode network --hardware-family "$family" --compute-lanes "$lanes" \
    --plm-bytes "$storage" --reuse-bytes "$reuse_bytes" --dma-width "$dma_width" \
    > "$out/engine-$name.log" 2>&1
done < "$out/engines.tsv"
python3 - "$out" <<'PY'
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);plan=json.loads((out/'network.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
artifacts={str(p.relative_to(out)):sha(p) for p in (out/'host.ll',out/'host-riscv.o',out/'parameters.bin',out/'network.json')}
for engine in plan['engines']:
    folder=out/'engines'/engine['name']
    for name in ('05_hw_acc.ll','architecture.xml','native_dma_contract.json'):
        artifacts[str((folder/name).relative_to(out))]=sha(folder/name)
(out/'compilation.json').write_text(json.dumps(dict(status='LOWERED_NOT_HARDWARE_VALIDATED',
    source='PyTorch/TOSA to compiler host and common LCS operator backend',
    engines=plan['engines'],sha256=artifacts),indent=2)+'\n')
PY
