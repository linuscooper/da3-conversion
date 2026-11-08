#!/usr/bin/env bash
# Orin stage: build engines, benchmark speed/memory/power, evaluate accuracy, write the report.
#   ./run_orin.sh setup | data | build | bench | eval | report | all
# Long run: start it in tmux or with  nohup ./run_orin.sh all > run.log 2>&1 &
# Every stage is resumable: finished engines and result rows are skipped on a re-run.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
SIZES="$SPEED_SIZES $EVAL_SIZE"
VENV=.venv
PY="$VENV/bin/python"

need_sudo() {
  sudo -v
  ( while true; do sudo -n true; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) 2>/dev/null &
}

stage_setup() {
  if [ -n "${CONDA_PREFIX:-}" ]; then echo "[warn] conda env active: run 'conda deactivate' first"; fi
  /usr/bin/python3 -c "import tensorrt; print('TensorRT', tensorrt.__version__)"
  /usr/bin/python3 -m venv --system-site-packages "$VENV"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements-orin.txt
  "$PY" -c "import tensorrt, cuda.bindings.runtime, cv2, h5py, scipy; print('[ok] Orin environment', tensorrt.__version__)"
  [ -x "$TRTEXEC" ] || { echo "trtexec not found at $TRTEXEC"; exit 1; }
}

stage_data() {
  "$PY" datasets.py --data-dir "$DATA_DIR" --get nyu diode
}

stage_build() {
  "$PY" orin_build.py --onnx-dir onnx --engines-dir engines --precisions $PRECISIONS --sizes $SIZES \
        --models $MODELS --trtexec "$TRTEXEC" --workspace-mb "$WORKSPACE_MB"
}

stage_bench() {
  need_sudo
  sudo nvpmodel -m 0 || true
  sudo jetson_clocks || true
  "$PY" orin_bench.py --engines-dir engines --seconds "$BENCH_SECONDS" --repeats "$BENCH_REPEATS" \
        --trtexec "$TRTEXEC" --rail "$POWER_RAIL" --models $MODELS --sizes $SIZES --precisions $PRECISIONS
}

stage_eval() {
  "$PY" orin_eval.py --engines-dir engines --size "$EVAL_SIZE" --models $MODELS --precisions $PRECISIONS \
        --data-dir "$DATA_DIR" --datasets $DATASETS --limit "$EVAL_LIMIT" --align "$ALIGN"
}

stage_report() {
  "$PY" report.py --results results
  cat results/summary.md
}

case "${1:-all}" in
  setup) stage_setup ;;
  data) stage_data ;;
  build) stage_build ;;
  bench) stage_bench ;;
  eval) stage_eval ;;
  report) stage_report ;;
  all) stage_setup; stage_data; stage_build; stage_bench; stage_eval; stage_report ;;
  *) echo "usage: $0 setup|data|build|bench|eval|report|all"; exit 1 ;;
esac
