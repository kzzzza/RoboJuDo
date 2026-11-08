"""Optional local inference caches. Source checkpoints are never rewritten."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_CACHE = Path(__file__).resolve().parents[2] / ".local/vision_accel"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(bundle_root):
    root = Path(bundle_root)
    names = {"models/hmr2_encoder.pt", "models/tcam_encoder.pt", "models/yolov8x.pt",
             "package.json", "pipeline.py"}
    for directory in ("code/lib/model", "code/lib/vendor/hmr2"):
        names.update(str(path.relative_to(root)) for path in (root / directory).rglob("*.py"))
    return {name: sha256(root / name) for name in sorted(names)}


def gpu_identity(device):
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    return dict(name=torch.cuda.get_device_name(index), capability=list(torch.cuda.get_device_capability(index)),
                device=f"cuda:{index}")


def validated_cache(directory, bundle_root, device):
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1 or manifest.get("sources") != fingerprint(bundle_root):
        raise RuntimeError("Visual acceleration cache does not match source weights; rebuild it")
    if manifest.get("torch") != torch.__version__ or manifest.get("gpu") != gpu_identity(device):
        raise RuntimeError("Visual acceleration cache belongs to another Torch/GPU/device; rebuild it")
    if not {"tcam_plain.pt", "tcam_context.pt"}.issubset(manifest.get("artifacts", {})):
        raise RuntimeError("Incomplete visual acceleration cache; rebuild it")
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise RuntimeError(f"Invalid visual acceleration artifact: {name}")
    return manifest


class TensorRTVisual:
    """Static HMR2 inference on an owned stream, synchronized with the caller."""

    def __init__(self, path, device, expected_version):
        import tensorrt as trt

        if trt.__version__ != expected_version:
            raise RuntimeError("TensorRT cache version differs from installed runtime; rebuild it")
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with torch.cuda.device(self.device):
            self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError("Cannot deserialize HMR2 TensorRT engine")
        with torch.cuda.device(self.device):
            self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        outputs = [name for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected exactly one HMR2 input and output")
        self.input_name, self.output_name = inputs[0], outputs[0]
        if tuple(self.engine.get_tensor_shape(self.input_name)) != (1, 3, 256, 256):
            raise ValueError("Unexpected HMR2 engine input")
        if tuple(self.engine.get_tensor_shape(self.output_name)) != (1, 1024):
            raise ValueError("Unexpected HMR2 engine output")
        types = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        self.input_dtype = types[self.engine.get_tensor_dtype(self.input_name)]
        self.output = torch.empty((1, 1024), device=self.device,
                                  dtype=types[self.engine.get_tensor_dtype(self.output_name)])
        self.context.set_tensor_address(self.output_name, self.output.data_ptr())
        self._input = None
        self.stream = torch.cuda.Stream(device=self.device)

    def __call__(self, batch):
        # Retain input until the next call; tensors must outlive asynchronous enqueue.
        self._input = batch["img"].to(device=self.device, dtype=self.input_dtype).contiguous()
        self.context.set_tensor_address(self.input_name, self._input.data_ptr())
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("HMR2 TensorRT execution failed")
        current.wait_stream(self.stream)
        # The reusable engine output must not alias tokens retained by later windows.
        return self.output.float().clone()


class HalfVisual:
    def __init__(self, model):
        self.model = model.half().eval()

    def __call__(self, batch):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.model(batch).float()


def trt_detector_backend(path, device):
    """Ultralytics 8.3 needs refreshed output addresses after dynamic buffer growth."""
    from ultralytics.nn.autobackend import AutoBackend

    class Backend(AutoBackend):
        def forward(self, image, *args, **kwargs):
            image = image.to(device=self.device, dtype=torch.float16 if self.fp16 else torch.float32).contiguous()
            if not self.context.set_input_shape("images", tuple(image.shape)):
                raise ValueError(f"Unsupported detector input shape: {tuple(image.shape)}")
            self.context.set_tensor_address("images", image.data_ptr())
            outputs = []
            for name in sorted(self.output_names):
                shape = tuple(self.context.get_tensor_shape(name))
                if any(size < 0 for size in shape):
                    raise ValueError(f"Unresolved detector output: {shape}")
                output = self.bindings[name].data
                if tuple(output.shape) != shape:
                    output.resize_(shape)
                # resize_ can reallocate: update the pointer on EVERY invocation.
                self.context.set_tensor_address(name, output.data_ptr())
                outputs.append(output)
            self._input = image  # retain storage until the next call
            current = torch.cuda.current_stream(self.device)
            self._execution_stream.wait_stream(current)
            if not self.context.execute_async_v3(self._execution_stream.cuda_stream):
                raise RuntimeError("YOLOv8x TensorRT execution failed")
            current.wait_stream(self._execution_stream)
            return outputs[0] if len(outputs) == 1 else outputs

    backend = Backend(str(path), device=torch.device(device), fp16=True)
    backend._execution_stream = torch.cuda.Stream(device=torch.device(device))
    return backend


def trt_detector_predictor(device):
    from ultralytics.data.augment import LetterBox
    from ultralytics.models.yolo.detect import DetectionPredictor

    class Predictor(DetectionPredictor):
        def setup_model(self, model, verbose=True):
            self.model = trt_detector_backend(model, self.args.device)
            self.device = self.model.device
            self.args.half = self.model.fp16
            self.model.eval()

        def pre_transform(self, images):
            # 8.3 only enables rectangular letterbox for .pt. Our engine is dynamic,
            # so preserve the Torch path's padding rather than changing detector inputs.
            same_shapes = len({image.shape for image in images}) == 1
            letterbox = LetterBox(self.imgsz, auto=same_shapes, stride=self.model.stride)
            return [letterbox(image=image) for image in images]

    # This installed YOLO.predict API takes an instance, rather than a predictor class.
    return Predictor(overrides=dict(device=str(device), half=True, conf=.5, classes=[0],
                                    task="detect", mode="predict", save=False, batch=1, verbose=False))


class CachedTemporal:
    def __init__(self, directory, device, precision="fp16"):
        self.half = precision == "fp16"
        self.plain = torch.jit.load(str(Path(directory) / "tcam_plain.pt"), map_location=device).eval()
        self.contextual = torch.jit.load(str(Path(directory) / "tcam_context.pt"), map_location=device).eval()

    def __call__(self, x, context=None, mask=None):
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            if self.half:
                x = x.half()
                context = None if context is None else context.half()
            if context is None:
                return self.plain(x)
            return self.contextual(x, context, mask)


class TemporalExport(torch.nn.Module):
    def __init__(self, model, contextual):
        super().__init__()
        self.model = model
        self.contextual = contextual

    def forward(self, x, context=None, mask=None):
        kwargs = dict(context_kv={"img_kv": context}, window_mask=mask) if self.contextual else {}
        return self.model(x, return_motion_feature=True, **kwargs).reshape(50, 1024).float()


class VisualExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model({"img": images})


def agreement(reference, actual):
    reference, actual = reference.float(), actual.float()
    if reference.shape != actual.shape or not torch.isfinite(actual).all():
        raise ValueError("Nonfinite or incorrectly shaped accelerated output")
    return dict(cosine_min=float(torch.nn.functional.cosine_similarity(reference, actual, dim=-1).min()),
                relative_l2=float(torch.linalg.vector_norm(actual - reference) /
                                  torch.linalg.vector_norm(reference).clamp_min(1e-8)),
                absolute_max=float((actual - reference).abs().max()))


def cuda_benchmark(call, warmup=10, repeat=50):
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        values = []
        for _ in range(repeat):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
    return dict(p50_ms=float(np.percentile(values, 50)), p95_ms=float(np.percentile(values, 95)))
