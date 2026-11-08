#!/usr/bin/env python3
"""Benchmark every engine with trtexec while logging tegrastats; write results/speed.csv.

  python3 orin_bench.py --engines-dir engines --seconds 30 --repeats 3

Per engine and repeat: 3 s idle tegrastats sample, then trtexec (CUDA graph, spin-wait, no host copies)
with tegrastats running. Reported numbers are medians over repeats. Needs sudo for tegrastats
(run_orin.sh keeps sudo alive). Lock clocks first (nvpmodel -m 0, jetson_clocks).
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENGINE_RE  # noqa: E402
from parse_logs import parse_build_log, parse_run_log, parse_tegrastats, summarize_tegrastats  # noqa: E402

FIELDS = ["model", "size", "precision", "status", "fps", "gpu_ms_mean", "gpu_ms_median", "gpu_ms_p99",
          "throughput_qps", "engine_mib", "weights_mib", "activation_mib", "context_mib", "build_s",
          "ram_idle_mb", "ram_peak_mb", "ram_delta_mb", "power_idle_w", "power_w", "power_delta_w",
          "energy_mj_per_frame", "gpu_util_pct", "repeats", "power_rail"]


def tegrastats(cmd, *args):
    subprocess.call(cmd + list(args), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sample(cmd, logfile, seconds=None, run=None, interval=100):
    logfile.unlink(missing_ok=True)
    tegrastats(cmd, "--stop")
    tegrastats(cmd, "--interval", str(interval), "--start", "--logfile", str(logfile))
    time.sleep(0.5)
    rc = None
    if run is not None:
        rc = run()
    else:
        time.sleep(seconds)
    tegrastats(cmd, "--stop")
    time.sleep(0.3)
    text = logfile.read_text(errors="replace") if logfile.exists() else ""
    return rc, parse_tegrastats(text)


def med(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines-dir", default="engines")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--out", default="results/speed.csv")
    ap.add_argument("--trtexec", default="/usr/bin/trtexec")
    ap.add_argument("--tegrastats", default="sudo -n tegrastats", help="command prefix for tegrastats")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rail", default="VDD_IN", help="power rail for module power (check tegrastats output)")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--sizes", nargs="*")
    ap.add_argument("--precisions", nargs="*")
    ap.add_argument("--force", action="store_true", help="re-run engines already in the CSV")
    a = ap.parse_args()

    teg = a.tegrastats.split()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(a.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not a.force:
        with out.open() as f:
            done = {(r["model"], r["size"], r["precision"]) for r in csv.DictReader(f) if r["status"] == "ok"}

    engines = []
    for e in sorted(Path(a.engines_dir).glob("*.engine"), key=lambda p: p.stat().st_size):
        m = ENGINE_RE.match(e.name)
        if not m:
            continue
        key = (m["model"], f"{m['h']}x{m['w']}", m["prec"])
        if (a.models and key[0] not in a.models) or (a.sizes and key[1] not in a.sizes) \
                or (a.precisions and key[2] not in a.precisions) or key in done:
            continue
        engines.append((e, key))
    print(f"[bench] {len(engines)} engines, {a.repeats} x {a.seconds} s each")

    write_header = not out.exists() or a.force
    with out.open("w" if a.force else "a", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        for engine, (model, size, prec) in engines:
            name = engine.stem
            runs = []
            for r in range(a.repeats):
                _, idle = sample(teg, log_dir / f"teg_idle_{name}_{r}.log", seconds=3)
                run_log = log_dir / f"run_{name}_{r}.log"
                cmd = [a.trtexec, f"--loadEngine={engine}", "--useCudaGraph", "--useSpinWait", "--noDataTransfers",
                       f"--duration={a.seconds}", "--warmUp=2000"]

                def run():
                    with run_log.open("w") as f:
                        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)

                rc, busy = sample(teg, log_dir / f"teg_run_{name}_{r}.log", run=run)
                info = parse_run_log(run_log.read_text(errors="replace"))
                si, sb = summarize_tegrastats(idle, 0, 0, a.rail), summarize_tegrastats(busy, rail=a.rail)
                runs.append(dict(rc=rc, **info, ram_idle_mb=si.get("ram_median_mb"), ram_peak_mb=sb.get("ram_peak_mb"),
                                 power_idle_w=si.get("power_w"), power_w=sb.get("power_w"),
                                 gpu_util_pct=sb.get("gpu_util_pct"), rails=sb.get("rails_seen")))
                print(f"        {name} run {r + 1}: {info.get('gpu_ms_mean')} ms, power {sb.get('power_w')} W", flush=True)
            ok = [x for x in runs if x["rc"] == 0 and x["gpu_ms_mean"]]
            build_log = log_dir / f"build_{name}.log"
            b = parse_build_log(build_log.read_text(errors="replace")) if build_log.exists() else {}
            row = dict(model=model, size=size, precision=prec, status="ok" if ok else "failed", repeats=len(ok),
                       engine_mib=round(engine.stat().st_size / 2**20, 1), weights_mib=b.get("weights_mib"),
                       activation_mib=b.get("activation_mib"), build_s=b.get("build_s"), power_rail=a.rail)
            if ok:
                for k in ("gpu_ms_mean", "gpu_ms_median", "gpu_ms_p99", "throughput_qps", "context_mib",
                          "ram_idle_mb", "ram_peak_mb", "power_idle_w", "power_w", "gpu_util_pct"):
                    row[k] = med(x[k] for x in ok)
                row["fps"] = round(1000 / row["gpu_ms_mean"], 2)
                if row["ram_peak_mb"] is not None and row["ram_idle_mb"] is not None:
                    row["ram_delta_mb"] = round(row["ram_peak_mb"] - row["ram_idle_mb"], 1)
                if row["power_w"] is not None:
                    row["energy_mj_per_frame"] = round(row["power_w"] * row["gpu_ms_mean"], 1)
                    if row["power_idle_w"] is not None:
                        row["power_delta_w"] = round(row["power_w"] - row["power_idle_w"], 2)
                elif ok[0].get("rails") is not None:
                    print(f"[warn] rail {a.rail} not in tegrastats output; rails seen: {ok[0]['rails']}")
            w.writerow(row)
            fcsv.flush()
            with (log_dir / "bench.jsonl").open("a") as f:
                f.write(json.dumps({"engine": engine.name, "runs": runs}) + "\n")
            print(f"[ok]   {name}: {row.get('fps')} FPS, RAM +{row.get('ram_delta_mb')} MB, {row.get('power_w')} W",
                  flush=True)


if __name__ == "__main__":
    main()
