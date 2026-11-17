# Operators-mode GEMM: lowering, HLS, RTL and timeline probe

Drivers used for the operators-mode GEMM results in `docs/Exp2PipelinePlan.md`
(runs under `exp2/runs/`). Everything except `incontainer.py` runs inside the
review container: the compiler at `/workspace`, `exp2/` at `/work`, the
Experiment 2 release at `/exp2` (read-only).

```bash
O=examples/esp-validation/reusable-gemm/operators
python3 $O/incontainer.py python3 /workspace/$O/ops-lower.py <run> MINI gemm LCS_COMPUTE_LANES=64
python3 $O/incontainer.py bash /workspace/$O/ops-hls.sh /work/runs/<run>/cases/MINI/gemm/generated /work/runs/<run>/hardware
python3 $O/incontainer.py python3 /workspace/$O/ops-rtl.py /work/runs/<run>/hardware /work/runs/<run>/cases/MINI/gemm/capture /work/runs/<run>/rtl-MINI-gemm 0 50000
python3 $O/incontainer.py python3 /workspace/$O/probe-rtl.py /work/runs/<run>/hardware/bambu_syn <capture> /work/runs/<run>/probe-<case>
python3 $O/compute-phases.py <probe dir>/simulation-0.log <probe dir>/probe.json
```

| Script | Does |
|---|---|
| `incontainer.py` | Runs a command in the review container (image and mounts of `tools/review.py`) |
| `ops-lower.py` | Lowers a frozen fixture with `LCS_TILES=operators`, packed DMA64; host-native check; capture for RTL (`OPS_SRC` overrides the fixture root) |
| `ops-hls.sh` | Bambu HLS of a lowering (xc7vx485t-2ffg1761, 20 ns) into a new directory |
| `ops-rtl.py` | Packed DMA64 component testbench (`dma64_tb.sv`) at the given stalls |
| `make-nan-case.py` | MINI gemm with beta = 0 and C all NaN |
| `make-probe.py` | Adds a timeline probe to `dma64_tb.sv` from a Bambu `forward_kernel.v`: FIFO handshakes between task nodes, node start/done, DMA requests |
| `probe-rtl.py` | Runs the probed testbench |
| `compute-phases.py`, `phases.py`, `fifosum.py`, `timeline.py` | Phases of the compute task (waiting for A/B, matmul, waiting for C, epilogue), FIFO summaries and timelines from a probe log |
