#!/usr/bin/env python3
"""Download / verify / iterate the evaluation datasets.

  python3 datasets.py --data-dir ~/da3data --get nyu diode

NYU Depth v2: official labelled set (1449 images, 2.8 GB), standard 654-image test split, Eigen crop.
DIODE: validation split (2.6 GB): 325 indoor + 446 outdoor images, laser ground truth.
"""
import argparse
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

NYU_MAT = ("nyu_depth_v2_labeled.mat",
           "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
           "520609c519fba3ba5ac58c8fefcc3530")
NYU_SPLITS = ("splits.mat", "https://raw.githubusercontent.com/cleinc/bts/master/utils/splits.mat", None)
DIODE_VAL = ("val.tar.gz", "http://diode-dataset.s3.amazonaws.com/val.tar.gz", "5c895d09201b88973c8fe4552a67dd85")

# Camera intrinsics (pixels, at the dataset's native resolution)
NYU_K = dict(fx=518.8579, fy=519.4696, w=640, h=480)     # NYU toolbox camera_params
DIODE_K = dict(fx=886.81, fy=927.06, w=1024, h=768)       # diode-devkit intrinsics.txt
# Valid ground-truth ranges (metres). NYU: standard 1e-3..10. DIODE: our choice, see README.
RANGES = {"nyu": (1e-3, 10.0), "diode_indoor": (1e-3, 50.0), "diode_outdoor": (1e-3, 300.0)}


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(dest_dir, item):
    name, url, digest = item
    dest = Path(dest_dir) / name
    if not dest.exists() or (digest and dest.stat().st_size < 1024):
        print(f"[get] {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        rc = subprocess.call(["wget", "-c", "-O", str(tmp), url])
        if rc != 0:
            raise SystemExit(f"download failed: {url} (place {name} in {dest_dir} manually)")
        tmp.rename(dest)
    if digest:
        got = md5(dest)
        if got != digest:
            raise SystemExit(f"{dest}: md5 {got} != {digest}; delete it and retry")
    return dest


def get_nyu(data_dir):
    d = Path(data_dir) / "nyu"
    d.mkdir(parents=True, exist_ok=True)
    fetch(d, NYU_MAT)
    fetch(d, NYU_SPLITS)
    print(f"[ok] NYU in {d}")


def get_diode(data_dir):
    d = Path(data_dir) / "diode"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "val").is_dir():
        tgz = fetch(d, DIODE_VAL)
        print("[extract] DIODE val")
        with tarfile.open(tgz) as t:
            t.extractall(d, filter="data")
    print(f"[ok] DIODE in {d}")


def eigen_mask(h=480, w=640):
    m = np.zeros((h, w), bool)
    m[45:471, 41:601] = True
    return m


def iter_nyu(data_dir, limit=0):
    """Yields (name, rgb uint8 HxWx3, depth float32 HxW metres, valid mask, K)."""
    import h5py
    import scipy.io as sio

    d = Path(data_dir) / "nyu"
    test = sio.loadmat(d / "splits.mat")["testNdxs"].ravel().astype(int) - 1  # 1-based in the file
    if limit:
        test = test[:limit]
    crop = eigen_mask()
    lo, hi = RANGES["nyu"]
    with h5py.File(d / "nyu_depth_v2_labeled.mat", "r") as f:
        images, depths = f["images"], f["depths"]
        for i in test:
            rgb = np.ascontiguousarray(np.asarray(images[i]).transpose(2, 1, 0))  # (3,W,H) -> (H,W,3)
            gt = np.asarray(depths[i], dtype=np.float32).T                         # (W,H) -> (H,W)
            assert rgb.shape == (480, 640, 3) and gt.shape == (480, 640), (rgb.shape, gt.shape)
            yield f"nyu_{i:04d}", rgb, gt, crop & (gt > lo) & (gt < hi), NYU_K


def iter_diode(data_dir, scene_type, limit=0):
    import cv2

    root = Path(data_dir) / "diode" / "val" / scene_type
    pngs = sorted(p for p in root.rglob("*.png"))
    if not pngs:
        raise SystemExit(f"no DIODE images under {root}")
    if limit:
        pngs = pngs[:limit]
    lo, hi = RANGES["diode_indoor" if scene_type == "indoors" else "diode_outdoor"]
    for p in pngs:
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        gt = np.load(p.with_name(p.stem + "_depth.npy")).squeeze().astype(np.float32)
        valid = np.load(p.with_name(p.stem + "_depth_mask.npy")).squeeze() > 0
        yield p.stem, rgb, gt, valid & (gt > lo) & (gt < hi), DIODE_K


def iter_dataset(name, data_dir, limit=0):
    if name == "nyu":
        return iter_nyu(data_dir, limit)
    if name == "diode_indoor":
        return iter_diode(data_dir, "indoors", limit)
    if name == "diode_outdoor":
        return iter_diode(data_dir, "outdoor", limit)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--get", nargs="+", choices=["nyu", "diode"], default=["nyu", "diode"])
    a = ap.parse_args()
    if "nyu" in a.get:
        get_nyu(a.data_dir)
    if "diode" in a.get:
        get_diode(a.data_dir)


if __name__ == "__main__":
    sys.exit(main())
