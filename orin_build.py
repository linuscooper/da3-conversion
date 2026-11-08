#!/usr/bin/env python3
"""Build TensorRT engines for every ONNX x precision on the Orin. Resumable: existing engines are skipped.

  python3 orin_build.py --onnx-dir onnx --engines-dir engines --precisions fp32 fp16 int8 \
      --sizes 280x336 378x504 420x504 [--models DA3-SMALL ...]

Precisions:
  fp32  strict FP32 (--noTF32)               reference for accuracy drift
  tf32  TensorRT's default FP32 (TF32 matmuls on Ampere)
  fp16  --fp16
  int8  <model>_<HxW>.int8.onnx from pc_quantize.py, built with --best (Model Optimizer's recommendation)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ONNX_RE  # noqa: E402
from parse_logs import parse_build_log  # noqa: E402

FLAGS = {"fp32": ["--noTF32"], "tf32": [], "fp16": ["--fp16"], "int8": ["--best"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--engines-dir", default="engines")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16", "int8"], choices=list(FLAGS))
    ap.add_argument("--sizes", nargs="*", help="only these sizes (default: all found)")
    ap.add_argument("--models", nargs="*", help="only these models (default: all found)")
    ap.add_argument("--trtexec", default="/usr/bin/trtexec")
    ap.add_argument("--workspace-mb", type=int, default=4096)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    onnx_dir, eng_dir, log_dir = Path(a.onnx_dir), Path(a.engines_dir), Path(a.logs_dir)
    for d in (eng_dir, log_dir, Path("cache")):
        d.mkdir(parents=True, exist_ok=True)
    jobs = []
    for src in sorted(onnx_dir.glob("*.onnx")):
        m = ONNX_RE.match(src.name)
        if not m or m["int8"]:
            continue
        size = f"{m['h']}x{m['w']}"
        if (a.models and m["model"] not in a.models) or (a.sizes and size not in a.sizes):
            continue
        for prec in a.precisions:
            onnx = src.with_name(src.name[: -len(".onnx")] + ".int8.onnx") if prec == "int8" else src
            jobs.append((m["model"], size, prec, onnx))
    # small models first, so early results arrive quickly; Giant last
    jobs.sort(key=lambda j: (j[3].stat().st_size if j[3].exists() else 0, j[2]))

    done = failed = skipped = missing = 0
    for model, size, prec, onnx in jobs:
        name = f"{model}_{size}_{prec}"
        engine, log = eng_dir / f"{name}.engine", log_dir / f"build_{name}.log"
        if engine.exists():
            skipped += 1
            continue
        if not onnx.exists():
            print(f"[miss]  {name}: {onnx.name} not found")
            missing += 1
            continue
        cmd = [a.trtexec, f"--onnx={onnx}", f"--saveEngine={engine}", *FLAGS[prec],
               f"--memPoolSize=workspace:{a.workspace_mb}", "--timingCacheFile=cache/timing.cache",
               "--skipInference"]
        if a.dry_run:
            print(" ".join(cmd))
            continue
        print(f"[build] {name} ...", flush=True)
        t0 = time.time()
        with log.open("w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        info = parse_build_log(log.read_text(errors="replace"))
        ok = rc == 0 and engine.exists()
        if not ok:
            engine.unlink(missing_ok=True)
            failed += 1
        else:
            done += 1
        rec = dict(engine=engine.name, model=model, size=size, precision=prec, rc=rc, ok=ok,
                   wall_s=round(time.time() - t0, 1), **info)
        with (log_dir / "builds.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        status = f"ok in {rec['wall_s']} s, {info.get('engine_mib')} MiB" if ok else f"FAILED (rc={rc}), see {log}"
        print(f"        {status}", flush=True)
    print(f"[done] built {done}, skipped {skipped} existing, missing ONNX {missing}, failed {failed}")


if __name__ == "__main__":
    main()
