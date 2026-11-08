"""Shared helpers: naming, preprocessing, inference runners (TensorRT and ONNX Runtime)."""
import re
from pathlib import Path

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PATCH = 14

# <model>_<H>x<W>.onnx | <model>_<H>x<W>.int8.onnx | <model>_<H>x<W>_<precision>.engine
ENGINE_RE = re.compile(r"^(?P<model>.+)_(?P<h>\d+)x(?P<w>\d+)_(?P<prec>[a-z0-9]+)\.engine$")
ONNX_RE = re.compile(r"^(?P<model>.+)_(?P<h>\d+)x(?P<w>\d+)(?P<int8>\.int8)?\.onnx$")


def parse_size(s):
    h, w = (int(v) for v in s.lower().split("x"))
    if h % PATCH or w % PATCH:
        raise ValueError(f"size {s}: height and width must be multiples of {PATCH}")
    return h, w


def is_metric(model_name):
    return "METRIC" in model_name.upper()


def preprocess(rgb_u8, h, w):
    """RGB uint8 HxWx3 -> float32 [1,3,h,w], resized (no aspect preservation) and ImageNet-normalised."""
    import cv2

    x = cv2.resize(rgb_u8, (w, h), interpolation=cv2.INTER_AREA if rgb_u8.shape[1] > w else cv2.INTER_LINEAR)
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class ORTRunner:
    """ONNX Runtime runner (used for testing and for FP32 references off-device)."""

    def __init__(self, path, providers=None):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
        i = self.sess.get_inputs()[0]
        self.inp = i.name
        self.h, self.w = int(i.shape[2]), int(i.shape[3])
        self.outputs = [o.name for o in self.sess.get_outputs()]

    def infer(self, x):
        return dict(zip(self.outputs, self.sess.run(None, {self.inp: x})))

    def close(self):
        self.sess = None


class TRTRunner:
    """TensorRT runner. Needs the system tensorrt module and cuda-python 13.x."""

    def __init__(self, path):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self.trt, self.cudart = trt, cudart
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"{path}: engine failed to load (built with another TensorRT version or device?)")
        self.ctx = self.engine.create_execution_context()
        self.stream = self._ck(cudart.cudaStreamCreate())
        self.host, self.dev, self.inputs, self.outputs = {}, {}, [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            self.host[name] = np.empty(shape, dtype=dtype)
            self.dev[name] = int(self._ck(cudart.cudaMalloc(self.host[name].nbytes)))
            self.ctx.set_tensor_address(name, self.dev[name])
            (self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs).append(name)
        self.inp = self.inputs[0]
        self.h, self.w = self.host[self.inp].shape[2:]

    def _ck(self, res):
        if int(res[0]) != 0:
            raise RuntimeError(f"CUDA error: {res[0]}")
        return res[1] if len(res) > 1 else None

    def infer(self, x):
        cu = self.cudart
        h2d, d2h = cu.cudaMemcpyKind.cudaMemcpyHostToDevice, cu.cudaMemcpyKind.cudaMemcpyDeviceToHost
        hin = self.host[self.inp]
        np.copyto(hin, x.astype(hin.dtype, copy=False))
        self._ck(cu.cudaMemcpyAsync(self.dev[self.inp], hin.ctypes.data, hin.nbytes, h2d, self.stream))
        if not self.ctx.execute_async_v3(int(self.stream)):
            raise RuntimeError("execute_async_v3 failed")
        for n in self.outputs:
            h = self.host[n]
            self._ck(cu.cudaMemcpyAsync(h.ctypes.data, self.dev[n], h.nbytes, d2h, self.stream))
        self._ck(cu.cudaStreamSynchronize(self.stream))
        return {n: self.host[n].copy() for n in self.outputs}

    def close(self):
        for p in self.dev.values():
            self.cudart.cudaFree(p)
        self.cudart.cudaStreamDestroy(self.stream)
        self.dev, self.ctx, self.engine = {}, None, None


def make_runner(path):
    path = Path(path)
    return TRTRunner(path) if path.suffix == ".engine" else ORTRunner(path)
