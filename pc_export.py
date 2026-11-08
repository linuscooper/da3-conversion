#!/usr/bin/env python3
"""Export Depth Anything 3 checkpoints to fixed-shape FP32 ONNX files for TensorRT.

  python3 pc_export.py --da3-src ~/Depth-Anything-3/src --out onnx \
      --models DA3-SMALL DA3-BASE DA3-LARGE-1.1 DA3-GIANT-1.1 DA3METRIC-LARGE \
      --sizes 280x336 378x504 420x504

Each model is traced once per size as a single-image model:
  input  : image [1,3,H,W] float32, RGB/255 then ImageNet mean/std
  outputs: depth [1,1,H,W] and conf (main series) or sky (metric/mono series), both [1,1,H,W]
The camera/ray branch and DA3's data-dependent sky post-processing are left out on purpose:
both are not needed for single-image depth and would be frozen incorrectly by tracing.
"""
import argparse
import gc
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import parse_size  # noqa: E402

HF_ORG = "depth-anything"
# Used only when a checkpoint's config.json has no "model_name" (assumption: -1.1 shares its base config).
CONFIG_FOR = {
    "DA3-SMALL": "da3-small", "DA3-BASE": "da3-base",
    "DA3-LARGE": "da3-large", "DA3-LARGE-1.1": "da3-large",
    "DA3-GIANT": "da3-giant", "DA3-GIANT-1.1": "da3-giant",
    "DA3METRIC-LARGE": "da3metric-large", "DA3MONO-LARGE": "da3mono-large",
}
UNUSED_MODULES = ("gs_head", "gs_adapter")
EXTERNAL_DATA_THRESHOLD = 1.8e9  # bytes of weights; above this, write one side-car .data file


def _cartesian_prod(a, b):
    """ONNX-friendly replacement for torch.cartesian_prod (used by DA3's 2D RoPE)."""
    ga, gb = torch.meshgrid(a, b, indexing="ij")
    return torch.stack([ga.reshape(-1), gb.reshape(-1)], dim=-1)


def build_net(name, ckpt, random_init):
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY

    cfg_name, ckpt_dir = CONFIG_FOR.get(name.upper()), None
    if not random_init:
        p = Path(ckpt) if ckpt else None
        if p is not None and p.is_dir():
            ckpt_dir = p
        else:
            from huggingface_hub import snapshot_download

            repo = ckpt or f"{HF_ORG}/{name}"
            ckpt_dir = Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"]))
        cfg_file = ckpt_dir / "config.json"
        if cfg_file.exists():
            cfg_name = json.loads(cfg_file.read_text()).get("model_name", cfg_name)
    if cfg_name is None:
        raise SystemExit(f"{name}: unknown model; add it to CONFIG_FOR")
    cfg = load_config(MODEL_REGISTRY[cfg_name])
    for k in UNUSED_MODULES:  # Gaussian-splatting branch (Giant): not needed for depth, needs extra deps
        if k in cfg:
            del cfg[k]
    net = create_object(cfg)
    if ckpt_dir is not None:
        from safetensors.torch import load_file

        files = sorted(ckpt_dir.glob("*.safetensors"))
        if not files:
            raise SystemExit(f"{name}: no .safetensors in {ckpt_dir}")
        sd = {}
        for f in files:
            sd.update(load_file(str(f)))
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if k.split(".", 1)[0] not in UNUSED_MODULES}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise SystemExit(f"{name}: checkpoint does not match config {cfg_name}: "
                             f"{len(missing)} missing (e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})")
    return net.float().eval(), cfg_name


class SingleImage(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image):
        h, w = image.shape[-2], image.shape[-1]
        feats, _ = self.net.backbone(image.unsqueeze(1), cam_token=None, export_feat_layers=[],
                                     ref_view_strategy="saddle_balanced")
        out = self.net.head(feats, h, w, patch_start_idx=0)
        second = out["depth_conf"] if "depth_conf" in out else out["sky"]
        return out["depth"][:, 0:1], second[:, 0:1]


def export_one(wrapper, second_name, h, w, out_path, opset, verify, fold=True):
    dummy = torch.randn(1, 3, h, w)
    with torch.no_grad():
        ref = [t.numpy() for t in wrapper(dummy)]
    with tempfile.TemporaryDirectory(dir=out_path.parent) as tmp:
        tmp_file = Path(tmp) / "model.onnx"
        with torch.no_grad():
            torch.onnx.export(wrapper, (dummy,), str(tmp_file), opset_version=opset, do_constant_folding=fold,
                              input_names=["image"], output_names=["depth", second_name], dynamo=False)
        import onnx

        model = onnx.load(str(tmp_file), load_external_data=True)
        weight_bytes = sum(int(np.prod(t.dims)) * 4 for t in model.graph.initializer)
        for f in (out_path, out_path.with_name(out_path.name + ".data")):
            f.unlink(missing_ok=True)
        if weight_bytes > EXTERNAL_DATA_THRESHOLD:
            onnx.save_model(model, str(out_path), save_as_external_data=True, all_tensors_to_one_file=True,
                            location=out_path.name + ".data", size_threshold=1024)
        else:
            onnx.save_model(model, str(out_path))
        del model
        gc.collect()
    import onnx

    onnx.checker.check_model(str(out_path))
    err = None
    if verify:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"image": dummy.numpy()})
        err = max(float(np.max(np.abs(g - r)) / (np.max(np.abs(r)) + 1e-12)) for g, r in zip(got, ref))
    return weight_bytes, err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--da3-src", required=True, help="path to Depth-Anything-3/src")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sizes", nargs="+", required=True, help="HxW, multiples of 14")
    ap.add_argument("--out", default="onnx")
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH_OR_REPO",
                    help="override checkpoint location for a model (default depth-anything/<NAME>)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-fold", action="store_true",
                    help="skip PyTorch constant folding (less RAM; TensorRT folds constants itself)")
    ap.add_argument("--no-verify", action="store_true", help="skip the ONNX Runtime vs PyTorch check")
    ap.add_argument("--random-init", action="store_true", help="test mode: random weights, no download")
    ap.add_argument("--force", action="store_true", help="re-export files that already exist")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.da3_src).expanduser().resolve()))
    torch.cartesian_prod = _cartesian_prod
    torch.set_grad_enabled(False)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = dict(kv.split("=", 1) for kv in a.ckpt)
    log = out_dir / "export_log.jsonl"
    sizes = [parse_size(s) for s in a.sizes]

    for name in a.models:
        todo = [(h, w) for h, w in sizes if a.force or not (out_dir / f"{name}_{h}x{w}.onnx").exists()]
        if not todo:
            print(f"[skip] {name}: all sizes exported")
            continue
        t0 = time.time()
        net, cfg_name = build_net(name, ckpts.get(name), a.random_init)
        wrapper = SingleImage(net).eval()
        params = sum(p.numel() for p in net.backbone.parameters()) + sum(p.numel() for p in net.head.parameters())
        second = "conf" if "depth_conf" in net.head(*_probe(net)) else "sky"
        print(f"[load] {name} ({cfg_name}): {params / 1e6:.1f}M params in backbone+head, outputs depth,{second} "
              f"({time.time() - t0:.0f}s)")
        for h, w in todo:
            out_path = out_dir / f"{name}_{h}x{w}.onnx"
            t1 = time.time()
            weight_bytes, err = export_one(wrapper, second, h, w, out_path, a.opset, not a.no_verify, not a.no_fold)
            rec = dict(model=name, config=cfg_name, size=f"{h}x{w}", onnx=out_path.name, params_m=round(params / 1e6, 2),
                       weights_mb=round(weight_bytes / 2**20, 1), second_output=second, opset=a.opset,
                       max_rel_err_vs_torch=err, export_s=round(time.time() - t1, 1), random_init=a.random_init,
                       peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2))
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[ok]   {out_path.name}: {rec['weights_mb']} MB, max rel err vs torch = {err}, {rec['export_s']} s, peak RAM {rec['peak_rss_gb']} GB")
        del wrapper, net


def _probe(net):
    """Tiny forward (28x28 input, 2x2 patches) to discover the head's output keys."""
    x = torch.zeros(1, 1, 3, 28, 28)
    feats, _ = net.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced")
    return feats, 28, 28, 0


if __name__ == "__main__":
    main()
