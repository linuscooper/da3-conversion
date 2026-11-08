#!/usr/bin/env python3
"""Build INT8 calibration sets: one .npz per input size, keyed by the ONNX input name ("image").

  python3 pc_calib.py --images ~/r1_frames --sizes 280x336 378x504 420x504 --n 300 --out calib

Use images from the robot's environment (R1 camera frames, undistorted, same preprocessing as deployment).
Never use the evaluation sets (NYU test split, DIODE val) here, or the accuracy numbers become optimistic.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import parse_size, preprocess  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder searched recursively for images")
    ap.add_argument("--sizes", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=300, help="number of calibration images (default 300)")
    ap.add_argument("--out", default="calib")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import cv2

    files = sorted(p for p in Path(a.images).expanduser().rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images under {a.images}")
    random.Random(a.seed).shuffle(files)
    files = files[: a.n]
    if len(files) < 100:
        print(f"[warn] only {len(files)} calibration images; INT8 accuracy may suffer")
    imgs = []
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is not None:
            imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in a.sizes:
        h, w = parse_size(s)
        arr = np.concatenate([preprocess(im, h, w) for im in imgs], axis=0).astype(np.float32)
        path = out / f"calib_{h}x{w}.npz"
        np.savez(path, image=arr)
        print(f"[ok] {path}: {arr.shape}, {arr.nbytes / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
