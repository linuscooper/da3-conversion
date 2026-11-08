#!/usr/bin/env python3
"""Merge results/speed.csv and results/accuracy.csv into results/summary.csv and results/summary.md."""
import argparse
import csv
from collections import OrderedDict
from pathlib import Path

PREC_ORDER = {"fp32": 0, "tf32": 1, "fp16": 2, "int8": 3, "onnx-fp32": 4, "onnx-int8": 5}


def read(path, key):
    rows = OrderedDict()
    if Path(path).exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                rows[key(r)] = r  # later rows win (re-runs)
    return rows


def num(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "–"


def sort_key(k):
    model, size, prec = k[:3]
    h, w = (int(x) for x in size.split("x"))
    return (h * w, model, PREC_ORDER.get(prec, 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    res = Path(a.results)
    speed = read(res / "speed.csv", lambda r: (r["model"], r["size"], r["precision"]))
    acc = read(res / "accuracy.csv", lambda r: (r["model"], r["size"], r["precision"], r["dataset"], r["mode"]))

    lines = ["# DA3 on Jetson Orin NX: benchmark summary", ""]
    if speed:
        lines += ["## Speed, memory, power (trtexec, GPU compute only)", "",
                  "| Model | Input | Precision | FPS | GPU ms mean | GPU ms p99 | Engine MiB | RAM +MB | Power W | mJ/frame |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for k in sorted(speed, key=sort_key):
            r = speed[k]
            if r["status"] != "ok":
                lines.append(f"| {k[0]} | {k[1]} | {k[2]} | failed | | | | | | |")
                continue
            lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {num(r['fps'], 1)} | {num(r['gpu_ms_mean'], 1)} | "
                         f"{num(r['gpu_ms_p99'], 1)} | {num(r['engine_mib'], 0)} | {num(r['ram_delta_mb'], 0)} | "
                         f"{num(r['power_w'], 1)} | {num(r['energy_mj_per_frame'], 0)} |")
        lines.append("")

    datasets = list(OrderedDict.fromkeys(k[3] for k in acc))
    engines = sorted({k[:3] for k in acc}, key=sort_key)
    if acc:
        head = "| Model | Input | Precision | " + " | ".join(f"{d} AbsRel | {d} δ1" for d in datasets) + \
               " | Drift vs FP32 | End-to-end ms |"
        lines += ["## Accuracy, scale-and-shift aligned (all models)", "", head,
                  "| --- | --- | --- | " + " | ".join("---: | ---:" for _ in datasets) + " | ---: | ---: |"]
        for e in engines:
            cells, drift, e2e = [], [], []
            for d in datasets:
                r = acc.get((*e, d, "aligned"))
                cells.append(f"{num(r['abs_rel'], 3)} | {num(r['delta1'], 3)}" if r else "– | –")
                if r:
                    drift.append(r["drift_vs_fp32"])
                    e2e.append(r["e2e_ms"])
            dv = [float(x) for x in drift if x not in (None, "")]
            ev = [float(x) for x in e2e if x not in (None, "")]
            lines.append(f"| {e[0]} | {e[1]} | {e[2]} | " + " | ".join(cells) +
                         f" | {num(sum(dv) / len(dv), 4) if dv else '–'} | {num(sum(ev) / len(ev), 1) if ev else '–'} |")
        lines.append("")
        metric_eng = [e for e in engines if any((*e, d, "raw") in acc for d in datasets)]
        if metric_eng:
            lines += ["## Metric accuracy, no fitting (metric model)", "",
                      "| Model | Input | Precision | " + " | ".join(f"{d} AbsRel | {d} δ1 | {d} RMSE m" for d in datasets)
                      + " |",
                      "| --- | --- | --- | " + " | ".join("---: | ---: | ---:" for _ in datasets) + " |"]
            for e in metric_eng:
                cells = []
                for d in datasets:
                    r = acc.get((*e, d, "raw"))
                    cells.append(f"{num(r['abs_rel'], 3)} | {num(r['delta1'], 3)} | {num(r['rmse'], 2)}" if r
                                 else "– | – | –")
                lines.append(f"| {e[0]} | {e[1]} | {e[2]} | " + " | ".join(cells) + " |")
            lines.append("")

    (res / "summary.md").write_text("\n".join(lines) + "\n")

    # wide CSV: one row per engine
    keys = sorted(set(speed) | set(engines), key=sort_key)
    cols = ["model", "size", "precision", "fps", "gpu_ms_mean", "gpu_ms_p99", "engine_mib", "ram_delta_mb",
            "power_w", "energy_mj_per_frame"]
    for d in datasets:
        cols += [f"{d}_aligned_abs_rel", f"{d}_aligned_delta1", f"{d}_raw_abs_rel", f"{d}_raw_delta1",
                 f"{d}_drift", f"{d}_e2e_ms"]
    with (res / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in keys:
            s = speed.get(k, {})
            row = dict(model=k[0], size=k[1], precision=k[2], **{c: s.get(c) for c in cols[3:10]})
            for d in datasets:
                al, raw = acc.get((*k, d, "aligned"), {}), acc.get((*k, d, "raw"), {})
                row.update({f"{d}_aligned_abs_rel": al.get("abs_rel"), f"{d}_aligned_delta1": al.get("delta1"),
                            f"{d}_raw_abs_rel": raw.get("abs_rel"), f"{d}_raw_delta1": raw.get("delta1"),
                            f"{d}_drift": al.get("drift_vs_fp32"), f"{d}_e2e_ms": al.get("e2e_ms")})
            w.writerow(row)
    print(f"[ok] {res / 'summary.md'} and {res / 'summary.csv'}")


if __name__ == "__main__":
    main()
