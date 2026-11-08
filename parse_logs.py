"""Parsers for trtexec build/run logs and tegrastats logs."""
import re
import statistics

_NUM = r"([0-9.]+)"


def _f(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def parse_build_log(text):
    return {
        "build_s": _f(r"Engine built in " + _NUM + " sec", text),
        "engine_mib": _f(r"Created engine with size: " + _NUM + " MiB", text),
        "weights_mib": _mib(_f(r"Total Weights Memory: (\d+) bytes", text)),
        "activation_mib": _mib(_f(r"Total Activation Memory: (\d+) bytes", text)),
        "build_peak_cpu_mib": _f(r"Peak memory usage of TRT CPU/GPU memory allocators: CPU (\d+) MiB", text),
        "build_peak_gpu_mib": _f(r"Peak memory usage of TRT CPU/GPU memory allocators: CPU \d+ MiB, GPU (\d+) MiB", text),
        "build_ok": "&&&& PASSED" in text,
    }


def _mib(b):
    return None if b is None else round(b / 2**20, 1)


def _stats(name, text):
    m = re.search(name + r": min = ([0-9.]+) ms, max = ([0-9.]+) ms, mean = ([0-9.]+) ms, median = ([0-9.]+) ms, "
                  r"percentile\(90%\) = ([0-9.]+) ms, percentile\(95%\) = ([0-9.]+) ms, percentile\(99%\) = ([0-9.]+) ms",
                  text)
    if not m:
        return {}
    keys = ["min", "max", "mean", "median", "p90", "p95", "p99"]
    return dict(zip(keys, (float(v) for v in m.groups())))


def parse_run_log(text):
    gpu = _stats("GPU Compute Time", text)
    host = _stats("Latency", text)
    return {
        "run_ok": "&&&& PASSED" in text,
        "throughput_qps": _f(r"Throughput: " + _NUM + " qps", text),
        "gpu_ms_mean": gpu.get("mean"), "gpu_ms_median": gpu.get("median"), "gpu_ms_p99": gpu.get("p99"),
        "host_ms_mean": host.get("mean"),
        "context_mib": _f(r"Created execution context with device memory size: " + _NUM + " MiB", text),
        "loaded_engine_mib": _f(r"Loaded engine size: " + _NUM + " MiB", text),
    }


_RAIL = re.compile(r"\b([A-Z0-9_]+) (\d+)mW/(\d+)mW")


def parse_tegrastats(text):
    """Return per-sample dicts: ram_mb, gpu_pct, rails {name: mW (instantaneous)}."""
    out = []
    for line in text.splitlines():
        ram = re.search(r"RAM (\d+)/(\d+)MB", line)
        if not ram:
            continue
        gpu = re.search(r"GR3D_FREQ (\d+)%", line)
        out.append({"ram_mb": int(ram.group(1)), "ram_total_mb": int(ram.group(2)),
                    "gpu_pct": int(gpu.group(1)) if gpu else None,
                    "rails": {k: int(v) for k, v, _ in _RAIL.findall(line)}})
    return out


def summarize_tegrastats(samples, trim_start=0.2, trim_end=0.05, rail="VDD_IN"):
    """Steady-state summary: drops the first/last part of the run (engine load, warm-up, teardown)."""
    if not samples:
        return {}
    n = len(samples)
    core = samples[int(n * trim_start): max(int(n * (1 - trim_end)), int(n * trim_start) + 1)] or samples
    power = [s["rails"].get(rail) for s in core if s["rails"].get(rail) is not None]
    gpu = [s["gpu_pct"] for s in core if s["gpu_pct"] is not None]
    return {
        "ram_peak_mb": max(s["ram_mb"] for s in samples),
        "ram_median_mb": statistics.median(s["ram_mb"] for s in core),
        "power_w": round(statistics.mean(power) / 1000, 2) if power else None,
        "gpu_util_pct": round(statistics.mean(gpu), 1) if gpu else None,
        "rails_seen": sorted({k for s in samples for k in s["rails"]}),
    }
