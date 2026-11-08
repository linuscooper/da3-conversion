#!/usr/bin/env python3
"""Accuracy and end-to-end timing of every engine on NYU and DIODE; writes results/accuracy.csv.

  python3 orin_eval.py --engines-dir engines --size 378x504 --data-dir ~/da3data

Scoring (per image on valid pixels, then averaged over images):
  aligned : prediction fitted to ground truth by least-squares scale + shift (all models; --align depth|disparity)
  raw     : metric model only, metres = output * focal_px_at_model_input / 300, no fitting
  median  : metric model only, raw output with a single median scale (separates scale from shape error)
Drift: mean |p - p_ref| / p_ref of raw outputs against the fp32 engine of the same model and size.
Also accepts .onnx files (ONNX Runtime) for testing off-device: --engines-dir onnx --pattern '*.onnx'.
"""
import argparse
import csv
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENGINE_RE, ONNX_RE, is_metric, make_runner, preprocess  # noqa: E402
from datasets import RANGES, iter_dataset  # noqa: E402

PREC_ORDER = ["fp32", "tf32", "fp16", "int8"]
METRICS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "delta1", "delta2", "delta3"]
FIELDS = ["model", "size", "precision", "dataset", "mode", "n_images", *METRICS,
          "drift_vs_fp32", "pre_ms", "infer_ms", "post_ms", "e2e_ms", "align"]


def depth_metrics(p, g):
    r = np.maximum(g / p, p / g)
    return dict(abs_rel=np.mean(np.abs(g - p) / g), sq_rel=np.mean((g - p) ** 2 / g),
                rmse=np.sqrt(np.mean((g - p) ** 2)), rmse_log=np.sqrt(np.mean((np.log(g) - np.log(p)) ** 2)),
                log10=np.mean(np.abs(np.log10(g) - np.log10(p))),
                delta1=np.mean(r < 1.25), delta2=np.mean(r < 1.25 ** 2), delta3=np.mean(r < 1.25 ** 3))


def fit_scale_shift(p, g):
    """Least-squares s, t minimising |s*p + t - g|^2 (closed form)."""
    pm, gm = p.mean(), g.mean()
    var = np.mean((p - pm) ** 2)
    s = np.mean((p - pm) * (g - gm)) / var if var > 0 else 1.0
    return s, gm - s * pm


def align(pred, gt, space):
    if space == "disparity":
        s, t = fit_scale_shift(1.0 / np.maximum(pred, 1e-6), 1.0 / gt)
        return 1.0 / np.maximum(s / np.maximum(pred, 1e-6) + t, 1e-6)
    s, t = fit_scale_shift(pred, gt)
    return s * pred + t


def list_models(a):
    groups = defaultdict(dict)
    for f in sorted(Path(a.engines_dir).glob(a.pattern)):
        m = ENGINE_RE.match(f.name)
        if m:
            key, prec = (m["model"], f"{m['h']}x{m['w']}"), m["prec"]
        else:
            m = ONNX_RE.match(f.name)
            if not m:
                continue
            key, prec = (m["model"], f"{m['h']}x{m['w']}"), ("onnx-int8" if m["int8"] else "onnx-fp32")
        if (a.size and key[1] != a.size) or (a.models and key[0] not in a.models) \
                or (a.precisions and prec not in a.precisions):
            continue
        groups[key][prec] = f
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines-dir", default="engines")
    ap.add_argument("--pattern", default="*.engine")
    ap.add_argument("--size", default="378x504", help="only engines of this input size ('' = all)")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--precisions", nargs="*")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=["nyu", "diode_indoor", "diode_outdoor"])
    ap.add_argument("--limit", type=int, default=0, help="images per dataset (0 = all)")
    ap.add_argument("--align", default="depth", choices=["depth", "disparity"])
    ap.add_argument("--drift-n", type=int, default=100, help="images per dataset used for drift vs fp32")
    ap.add_argument("--out", default="results/accuracy.csv")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import cv2

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not a.force:
        with out.open() as f:
            done = {(r["model"], r["size"], r["precision"], r["dataset"]) for r in csv.DictReader(f)}
    groups = list_models(a)
    if not groups:
        raise SystemExit(f"nothing to evaluate in {a.engines_dir} (pattern {a.pattern}, size {a.size})")
    new_file = not out.exists() or a.force
    fcsv = out.open("w" if a.force else "a", newline="")
    w = csv.DictWriter(fcsv, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    for (model, size), by_prec in sorted(groups.items()):
        metric = is_metric(model)
        # reference (fp32) first so drift can be computed for the others
        order = sorted(by_prec, key=lambda p: (p not in ("fp32", "onnx-fp32"),
                                               PREC_ORDER.index(p) if p in PREC_ORDER else len(PREC_ORDER), p))
        refs = {}  # dataset -> {image name: fp32 raw output at model resolution}, for drift
        is_ref = lambda pr: pr in ("fp32", "onnx-fp32")  # noqa: E731
        others_todo = {d for pr in order if not is_ref(pr) for d in a.datasets if (model, size, pr, d) not in done}
        for prec in order:
            path = by_prec[prec]
            todo = [d for d in a.datasets if (model, size, prec, d) not in done]
            # the fp32 engine also runs (on drift_n images only) when it is done but others still need references
            ref_only = [d for d in a.datasets if is_ref(prec) and d not in todo and d in others_todo]
            if not todo and not ref_only:
                continue
            print(f"[eval] {path.name}", flush=True)
            runner = make_runner(path)
            h, w_in = runner.h, runner.w
            x0 = np.zeros((1, 3, h, w_in), np.float32)
            for _ in range(3):
                runner.infer(x0)  # warm-up
            for ds in todo + ref_only:
                lo, hi = RANGES[ds]
                per, times, drift = defaultdict(list), defaultdict(list), []
                ref = refs.setdefault(ds, {})
                limit = a.drift_n if ds in ref_only else a.limit
                for name, rgb, gt, valid, K in iter_dataset(ds, a.data_dir, limit):
                    if valid.sum() < 100:
                        continue
                    t0 = time.perf_counter()
                    x = preprocess(rgb, h, w_in)
                    t1 = time.perf_counter()
                    raw = runner.infer(x)["depth"][0, 0].astype(np.float32)
                    t2 = time.perf_counter()
                    pred = cv2.resize(raw, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                    t3 = time.perf_counter()
                    times["pre"].append(t1 - t0)
                    times["infer"].append(t2 - t1)
                    times["post"].append(t3 - t2)
                    if is_ref(prec):
                        if len(ref) < a.drift_n:
                            ref[name] = raw
                    elif name in ref:
                        r = ref[name]
                        drift.append(float(np.mean(np.abs(raw - r) / (np.abs(r) + 1e-6))))
                    if ds in ref_only:
                        continue
                    g = gt[valid]
                    pv = np.maximum(pred[valid], 1e-6)
                    per["aligned"].append(depth_metrics(np.clip(align(pv, g, a.align), lo, hi), g))
                    if metric:
                        f_model = 0.5 * (K["fx"] * w_in / K["w"] + K["fy"] * h / K["h"])
                        pm = pv * f_model / 300.0
                        per["raw"].append(depth_metrics(np.clip(pm, lo, hi), g))
                        per["median"].append(depth_metrics(np.clip(pm * np.median(g) / np.median(pm), lo, hi), g))
                if ds in ref_only:
                    continue
                n = len(per["aligned"])
                tm = {k: round(1000 * float(np.mean(v[3:] if len(v) > 3 else v)), 2) for k, v in times.items()}
                for mode, rows in per.items():
                    row = dict(model=model, size=size, precision=prec, dataset=ds, mode=mode, n_images=n,
                               align=a.align if mode == "aligned" else "", pre_ms=tm.get("pre"),
                               infer_ms=tm.get("infer"), post_ms=tm.get("post"),
                               e2e_ms=round(sum(tm.values()), 2),
                               drift_vs_fp32=round(float(np.mean(drift)), 5) if drift else None)
                    for mk in METRICS:
                        row[mk] = round(float(np.mean([r[mk] for r in rows])), 4)
                    w.writerow(row)
                fcsv.flush()
                d1 = np.mean([r["delta1"] for r in per["aligned"]]) if n else float("nan")
                print(f"        {ds}: n={n} aligned delta1={d1:.3f}, inference {tm.get('infer')} ms", flush=True)
            runner.close()
            del runner
            gc.collect()
    fcsv.close()


if __name__ == "__main__":
    main()
