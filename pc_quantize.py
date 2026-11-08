#!/usr/bin/env python3
"""INT8 post-training quantization with NVIDIA Model Optimizer (explicit Q/DQ nodes in the ONNX).

  python3 pc_quantize.py --onnx-dir onnx --calib-dir calib [--method entropy|max] [--eps cuda:0 cpu]

For every <model>_<H>x<W>.onnx it writes <model>_<H>x<W>.int8.onnx, calibrated with calib/calib_<H>x<W>.npz.
Build the result on the Orin with:  trtexec --onnx=<file>.int8.onnx --best
"""
import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ONNX_RE  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--calib-dir", default="calib")
    ap.add_argument("--method", default="entropy", choices=["entropy", "max"])
    ap.add_argument("--eps", nargs="+", default=None,
                    help="calibration execution providers in order of preference, e.g. 'cuda:0 cpu' "
                         "(default: modelopt's own default)")
    ap.add_argument("--models", nargs="*", help="only these models (default: all ONNX files found)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import modelopt.onnx.quantization as moq

    accepted = set(inspect.signature(moq.quantize).parameters)
    onnx_dir = Path(a.onnx_dir)
    log = onnx_dir / "quantize_log.jsonl"
    for src in sorted(onnx_dir.glob("*.onnx")):
        m = ONNX_RE.match(src.name)
        if not m or m["int8"] or (a.models and m["model"] not in a.models):
            continue
        dst = src.with_name(src.name[: -len(".onnx")] + ".int8.onnx")
        if dst.exists() and not a.force:
            print(f"[skip] {dst.name} exists")
            continue
        calib = Path(a.calib_dir) / f"calib_{m['h']}x{m['w']}.npz"
        if not calib.exists():
            print(f"[miss] {calib} not found; skipping {src.name}")
            continue
        data = dict(np.load(calib))
        external = src.with_name(src.name + ".data").exists()
        kwargs = dict(onnx_path=str(src), quantize_mode="int8", calibration_data=data,
                      calibration_method=a.method, output_path=str(dst),
                      use_external_data_format=external, high_precision_dtype="fp16")
        if a.eps:
            kwargs["calibration_eps"] = a.eps
        dropped = sorted(k for k in kwargs if k not in accepted)
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        if dropped:
            print(f"[note] this modelopt version ignores: {dropped}")
        t0 = time.time()
        print(f"[int8] {src.name} with {calib.name} ({data['image'].shape[0]} images, {a.method})")
        moq.quantize(**kwargs)
        rec = dict(onnx=src.name, int8=dst.name, calib=calib.name, n_calib=int(data["image"].shape[0]),
                   method=a.method, seconds=round(time.time() - t0, 1))
        with log.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[ok]   {dst.name} in {rec['seconds']} s")


if __name__ == "__main__":
    main()
