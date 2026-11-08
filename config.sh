# da3bench configuration, read by run_pc.sh and run_orin.sh. Edit, then run.

# Models to benchmark (Hugging Face names under depth-anything/)
MODELS="DA3-SMALL DA3-BASE DA3-LARGE-1.1 DA3-GIANT-1.1 DA3METRIC-LARGE"

# Input sizes (H x W, multiples of 14)
SPEED_SIZES="280x336 420x504"   # 1.2:1, the R1 camera's aspect ratio: speed, memory, power only
EVAL_SIZE="378x504"             # 4:3, matches NYU and DIODE: speed + accuracy

# Engine precisions: fp32 (strict, --noTF32), tf32 (TensorRT default FP32), fp16, int8
PRECISIONS="fp32 fp16 int8"

# ---- PC (export + INT8 calibration) ----
DA3_SRC="$HOME/Depth-Anything-3/src"   # git clone https://github.com/ByteDance-Seed/Depth-Anything-3
CALIB_IMAGES="$HOME/r1_frames"         # robot-environment images for INT8 calibration (not the test sets)
CALIB_N=300
CALIB_METHOD="entropy"                 # entropy | max
CALIB_EPS="cuda:0 cpu"                 # calibration devices, in order of preference
ORIN=""                                # e.g. orinnx@192.168.1.50 to rsync ONNX files; empty = copy yourself
ORIN_DIR="~/da3bench"

# ---- Orin (build + benchmark + evaluation) ----
TRTEXEC="/usr/bin/trtexec"
WORKSPACE_MB=4096
BENCH_SECONDS=30
BENCH_REPEATS=3
POWER_RAIL="VDD_IN"                    # whole-module input power in tegrastats on Orin NX
DATA_DIR="$HOME/da3data"
DATASETS="nyu diode_indoor diode_outdoor"
EVAL_LIMIT=0                           # images per dataset, 0 = all (try 20 for a quick end-to-end check)
ALIGN="depth"                          # depth | disparity: space for the scale-and-shift fit
