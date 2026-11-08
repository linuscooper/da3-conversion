# da3bench: Depth Anything 3 on Jetson Orin NX

Exports DA3 models to ONNX, builds TensorRT engines in FP32, FP16 and INT8, and measures each engine's speed, memory, power and accuracy. The result is one table.

```
PC   ./run_pc.sh all    export ONNX -> INT8 calibration -> quantize -> copy to Orin
Orin ./run_orin.sh all  setup -> datasets -> build engines -> benchmark -> evaluate -> report
```

Every stage resumes where it stopped: finished engines and result rows are skipped. Run single stages with `./run_orin.sh build`, `bench`, `eval` or `report`.

## What you get

`results/summary.md` and `results/summary.csv` have one row per model × input size × precision. Speed, memory and power cover every engine:

| Column | Source |
| --- | --- |
| FPS, GPU ms mean / p99 | `trtexec` with CUDA graph, spin-wait and no host copies (30 s, median of 3 runs) |
| Engine MiB, weights and activation MiB | engine file and TensorRT build log |
| RAM +MB | tegrastats peak RAM during the run minus idle RAM (on Jetson, GPU memory is system RAM) |
| Power W, mJ/frame | tegrastats `VDD_IN` (whole module) during the run; energy = power × GPU time |

Accuracy columns are filled at the evaluation size (378×504):

| Column | Source |
| --- | --- |
| AbsRel, δ1 | per dataset: NYU v2 test (654), DIODE val indoor (325) and outdoor (446) |
| Drift vs FP32 | mean \|p − p_fp32\| / p_fp32 of raw outputs on 100 images per dataset |
| End-to-end ms | Python pipeline: resize + normalise, H2D + inference + D2H, upscale |

`results/speed.csv` and `results/accuracy.csv` have every field, including RMSE, δ2, δ3 and per-stage times.

## 1. PC: export and INT8 calibration

You need Linux x86-64 and Python 3.10–3.13. An NVIDIA GPU is recommended, because INT8 calibration on the CPU is slow. Export peaks at about 4.5× a model's weights in RAM:

| Model | FP32 weights | Export RAM |
| --- | --- | --- |
| Small | 0.1 GB | 1.3 GB (measured) |
| Base | 0.4 GB | about 2 GB |
| Large and Metric-Large | 1.3 GB | 5.7 GB (Metric-Large, measured) |
| Giant | about 4.4 GB | about 20 GB, so use a 32 GB machine |

Disk: about 30 GB for all ONNX files (estimate).

1. Edit `config.sh`:
   - `MODELS`, `SPEED_SIZES`, `EVAL_SIZE`, `PRECISIONS`
   - `DA3_SRC`: where to clone the DA3 repository
   - `CALIB_IMAGES`: a folder of 300+ images of the robot's environment, such as undistorted R1 frames. Do not use the test sets.
   - `ORIN`: `user@ip`, so the ONNX files are copied over automatically.
2. Run `./run_pc.sh all`. Checkpoints download from Hugging Face (`depth-anything/<MODEL>`).

The export traces each model as a single-image network. It also verifies each ONNX file against PyTorch:
- **Input:** `image` [1,3,H,W]: RGB/255, normalised with the ImageNet mean and std.
- **Outputs:** `depth` [1,1,H,W] plus `conf` (main series) or `sky` (metric series).
- **Removed on purpose:**
  - The camera/ray branch, which single-image depth doesn't need.
  - DA3's data-dependent sky post-processing, which tracing would bake into the graph.
  - Giant's Gaussian-splatting head.

  Speeds are therefore for depth only.

## 2. Orin: build, benchmark, evaluate

Run `conda deactivate`, then start the run inside tmux: `./run_orin.sh all`. `sudo` is needed for `tegrastats`, `nvpmodel` and `jetson_clocks`. The script asks once and keeps sudo alive.

Space needed (estimates):
- Engines: about 40 GB for 5 models × 3 sizes × 3 precisions. Drop `fp32` from `PRECISIONS`, or use fewer sizes, to cut this.
- Datasets: about 9 GB in `DATA_DIR`.

Time (estimate): a Large FP16 build at 280×504 took 457 s on your Orin. 45 builds, plus benchmarks and evaluation, is roughly a night to a day.

Do a quick trial first: set `MODELS="DA3-SMALL"`, `PRECISIONS="fp16"` and `EVAL_LIMIT=20`, then run `./run_orin.sh all`.

Precisions:
- `fp32`: strict FP32 (`--noTF32`). This is the accuracy reference.
- `tf32`: TensorRT's default FP32, which uses TF32 for matrix multiplies.
- `fp16`: `--fp16`.
- `int8`: Model Optimizer Q/DQ ONNX built with `--best`.

FP8 is not available on Orin. DLA is not supported for Q/DQ models between TensorRT 10.7 and 11.3, and you have 10.16.

## Scoring rules

- **aligned** (all models): per image, a least-squares scale and shift fitted to ground truth, in depth space (`ALIGN=disparity` to switch). This follows the DA2-style protocol the DA3 paper cites. The paper excerpt I could read doesn't say which space it uses.
- **raw** (metric model): metres = output × focal / 300, with the focal length rescaled to the model input resolution, as in DA3's `utils/alignment.py`. There is no fitting. **median** adds one median scale, to separate scale error from shape error.
- **Valid pixels:**
  - NYU: Eigen crop, 0.001–10 m.
  - DIODE: dataset mask, with 50 m indoor and 300 m outdoor caps. The caps are my choice.
- **Intrinsics:**
  - NYU: fx 518.86, fy 519.47 (NYU toolbox).
  - DIODE: fx 886.81, fy 927.06 (diode-devkit).
- **Sanity check:** the DA3 paper reports δ1 of 97.4 on NYU and 95.4 on DIODE for its best model, under its own protocol and resolution.

## Tested and not tested

Tested in a CPU sandbox with randomly initialised weights, because Hugging Face downloads were blocked there:
- ONNX export with verification against PyTorch (max relative error ≤ 2e-7) for Small, Base, Metric-Large, and a 4-block Giant with the same layer types. Main-series Large ran out of memory on the 8 GB sandbox.
- INT8 quantization with Model Optimizer 0.48 (380 Q/DQ pairs; runs in ONNX Runtime).
- The evaluation pipeline on synthetic data in the real NYU (.mat/HDF5) and DIODE layouts, including drift and resume.
- Build, benchmark and report parsing against stand-ins that print your real `trtexec` 10.16 and `tegrastats` log format.

Not tested:
- Loading the real checkpoints. The script stops if the weights don't match the config.
- TensorRT builds of these ONNX files.
- The `tegrastats` rail names on your device. The benchmark warns if `VDD_IN` is missing; set `POWER_RAIL`.
- Downloads from the NYU server, which is sometimes slow or down. You can place `nyu_depth_v2_labeled.mat` in `DATA_DIR/nyu` manually.

## Licences

DA3-Small, Base and Metric-Large are Apache 2.0. DA3-Large-1.1 and Giant-1.1 are CC BY-NC 4.0, which allows benchmarking but not commercial use.
