#!/usr/bin/env bash
# PC stage: export FP32 ONNX for every model x size, build INT8 calibration sets, quantize, copy to the Orin.
#   ./run_pc.sh setup | export | calib | quantize | push | all
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
SIZES="$SPEED_SIZES $EVAL_SIZE"
VENV=.venv-pc

stage_setup() {
  [ -d "$DA3_SRC/depth_anything_3" ] || git clone https://github.com/ByteDance-Seed/Depth-Anything-3 "$(dirname "$DA3_SRC")"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r requirements-pc.txt
  echo "[ok] PC environment in $VENV"
}

stage_export() {
  "$VENV/bin/python" -W ignore pc_export.py --da3-src "$DA3_SRC" --models $MODELS --sizes $SIZES --out onnx
}

stage_calib() {
  "$VENV/bin/python" pc_calib.py --images "$CALIB_IMAGES" --sizes $SIZES --n "$CALIB_N" --out calib
}

stage_quantize() {
  if [[ " $PRECISIONS " != *" int8 "* ]]; then echo "[skip] int8 not in PRECISIONS"; return; fi
  "$VENV/bin/python" pc_quantize.py --onnx-dir onnx --calib-dir calib --method "$CALIB_METHOD" --eps $CALIB_EPS --models $MODELS
}

stage_push() {
  if [ -z "$ORIN" ]; then echo "[skip] ORIN not set: copy onnx/ to the Orin's da3bench/onnx yourself"; return; fi
  ssh "$ORIN" "mkdir -p $ORIN_DIR/onnx"
  rsync -avP --exclude '*.jsonl' onnx/ "$ORIN:$ORIN_DIR/onnx/"
  rsync -avP --exclude 'onnx' --exclude 'calib' --exclude '.venv*' --exclude 'engines' --exclude 'results' \
        --exclude 'logs' ./ "$ORIN:$ORIN_DIR/"
}

case "${1:-all}" in
  setup) stage_setup ;;
  export) stage_export ;;
  calib) stage_calib ;;
  quantize) stage_quantize ;;
  push) stage_push ;;
  all) stage_setup; stage_export; stage_calib; stage_quantize; stage_push ;;
  *) echo "usage: $0 setup|export|calib|quantize|push|all"; exit 1 ;;
esac
