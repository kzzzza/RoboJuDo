"""Build GPU-specific caches from the existing visual weights, without robot access."""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PYAV_LOGGING", "off")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for variable, relative in (("XDG_CACHE_HOME", ".local/cache"), ("TORCH_HOME", ".local/cache/torch"),
                           ("HF_HOME", ".local/cache/huggingface"), ("CUDA_CACHE_PATH", ".local/cache/cuda"),
                           ("TMPDIR", ".local/setup"), ("MPLCONFIGDIR", ".local/video_runtime"),
                           ("YOLO_CONFIG_DIR", ".local/video_runtime")):
    (ROOT / relative).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(ROOT / relative))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import torch

from robojudo.video.vision_accel import (
    DEFAULT_CACHE, TemporalExport, TensorRTVisual, VisualExport,
    agreement, cuda_benchmark, fingerprint, gpu_identity, sha256,
    trt_detector_backend,
)
from robojudo.video.vision_worker import _load_bundle


def build_detector(args, root, directory, manifest):
    import tensorrt as trt
    import ultralytics
    from ultralytics import YOLO

    detector = YOLO(str(root / "models/yolov8x.pt"))
    # Exporter chooses its output from pt_path; keep all generated files in the cache.
    detector.model.pt_path = str(directory / "yolov8x.pt")
    export = torch.onnx.export

    def legacy_export(*positional, **kwargs):
        kwargs["dynamo"] = False  # this installed Ultralytics predates Torch's new export default
        # Ultralytics 8.3 forces dynamic export to CPU; Torch 2.9 supports CUDA tracing.
        positional = list(positional)
        positional[0] = positional[0].to(args.device)
        positional[1] = positional[1].to(args.device)
        return export(*positional, **kwargs)

    with patch.object(torch.onnx, "export", side_effect=legacy_export):
        onnx_path = detector.export(format="onnx", device=args.device, half=False, dynamic=True,
                                    imgsz=640, batch=1, opset=17, simplify=False)
    if not onnx_path or not Path(onnx_path).is_file():
        raise RuntimeError("YOLOv8x ONNX export failed")
    import onnx
    metadata = {item.key: item.value for item in onnx.load(onnx_path).metadata_props}
    # Ultralytics' loader expects numerical metadata, rather than ONNX string properties.
    import ast
    for key in ("stride", "batch", "imgsz", "names"):
        metadata[key] = ast.literal_eval(metadata[key])
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Keep dynamic anchor Range/decode in FP32; TensorRT does not accept FP16 Range.
    # Use FP16 convolution tactics and FP16 input, with FP32 where required by operators.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    network.get_input(0).dtype = trt.float16
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
    config.builder_optimization_level = 3
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, 256, 256), (1, 3, 480, 640), (1, 3, 640, 640))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("YOLOv8x TensorRT build failed")
    engine_path = directory / "yolov8x.engine"
    header = json.dumps(metadata).encode()
    with engine_path.open("wb") as stream:
        stream.write(len(header).to_bytes(4, byteorder="little"))
        stream.write(header)
        stream.write(bytes(plan))
    del plan, parser, network, config, builder
    gc.collect()
    torch.cuda.empty_cache()
    engine = trt_detector_backend(engine_path, args.device)
    reference = detector.model.to(args.device).eval().fuse().half()
    checks = []
    with torch.inference_mode():
        for height, width in ((480, 640), (640, 480), (384, 640), (640, 640)):
            image = torch.rand(1, 3, height, width, device=args.device, dtype=torch.float16)
            ref = reference(image)[0]
            actual = engine(image)
            check = agreement(ref.transpose(1, 2), actual.transpose(1, 2))
            if check["cosine_min"] < .999 or check["relative_l2"] > .03:
                raise RuntimeError(f"YOLOv8x TensorRT validation failed: {check}")
            checks.append(check)
        manifest["validation"]["yolov8x"] = dict(agreement=checks,
            torch=cuda_benchmark(lambda: reference(image)), tensorrt=cuda_benchmark(lambda: engine(image)))
    manifest["tensorrt"] = trt.__version__
    manifest["ultralytics"] = ultralytics.__version__
    manifest["artifacts"]["yolov8x.engine"] = sha256(engine_path)
    print("yolov8x", manifest["validation"]["yolov8x"], flush=True)


def build(args):
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    torch.manual_seed(3101)
    root = Path(args.packages).resolve() / "original_pipeline"
    directory = Path(args.cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # An interrupted rebuild must not leave an older manifest approving new files.
    (directory / "manifest.json").unlink(missing_ok=True)
    bundle = _load_bundle(root)
    manifest = dict(version=1, sources=fingerprint(root), torch=torch.__version__,
                    gpu=gpu_identity(args.device), artifacts={}, validation={},
                    temporal_precision=args.temporal_precision)
    frontend = bundle.load_frontend(args.device)
    x = torch.randn(1, 50, 1024, device=args.device)
    context = torch.randn(1, 16, 1024, device=args.device)
    mask = torch.zeros(1, 16, dtype=torch.bool, device=args.device)
    with torch.inference_mode():
        references = {}
        for contextual in (False, True):
            reference = TemporalExport(frontend, contextual).eval()
            references[contextual] = []
            for valid in (0, 1, 3, 8):
                mask[:] = False
                mask[:, :valid] = True
                inputs = (x, context, mask) if contextual else (x,)
                references[contextual].append(reference(*inputs).clone())
        if args.temporal_precision == "fp16":
            frontend.half()
            x, context = x.half(), context.half()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                               enabled=args.temporal_precision == "fp16", cache_enabled=False):
        for contextual, name in ((False, "tcam_plain.pt"), (True, "tcam_context.pt")):
            wrapper = TemporalExport(frontend, contextual).eval()
            inputs = (x, context, mask) if contextual else (x,)
            traced = torch.jit.trace(wrapper, inputs, strict=False, check_trace=False).eval()
            checks = []
            for i, valid in enumerate((0, 1, 3, 8)):
                mask[:] = False
                mask[:, :valid] = True
                ref = references[contextual][i]
                check = agreement(ref, traced(*inputs))
                if check["cosine_min"] < .9999 or check["relative_l2"] > .01:
                    raise RuntimeError(f"TCAM trace validation failed: {check}")
                checks.append(check)
            traced.save(str(directory / name))
            manifest["artifacts"][name] = sha256(directory / name)
            manifest["validation"][name] = dict(agreement=checks,
                eager=cuda_benchmark(lambda: wrapper(*inputs)), traced=cuda_benchmark(lambda: traced(*inputs)))
            print(name, manifest["validation"][name], flush=True)
    del traced, wrapper, reference, frontend, references, x, context, mask
    gc.collect()
    torch.cuda.empty_cache()

    if args.hmr_backend == "tensorrt":
        import tensorrt as trt

        visual = bundle.load_visual(args.device).half()
        images = torch.randn(1, 3, 256, 256, device=args.device, dtype=torch.float16)
        onnx_path = directory / "hmr2_fp16.onnx"
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # Legacy export retains the exact existing module operations and weights.
            torch.onnx.export(VisualExport(visual).eval(), (images,), str(onnx_path),
                              input_names=["images"], output_names=["features"],
                              opset_version=17, dynamo=False, do_constant_folding=True)
        del visual
        gc.collect()
        torch.cuda.empty_cache()
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb << 20)
        config.builder_optimization_level = 3
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("HMR2 TensorRT engine build failed")
        engine_path = directory / "hmr2.engine"
        engine_path.write_bytes(bytes(plan))
        del plan, parser, network, builder, config
        gc.collect()
        torch.cuda.empty_cache()
        engine = TensorRTVisual(engine_path, args.device, trt.__version__)
        visual = bundle.load_visual(args.device)
        checks = []
        with torch.inference_mode():
            for _ in range(5):
                image = torch.randn(1, 3, 256, 256, device=args.device)
                ref = visual({"img": image})
                check = agreement(ref, engine({"img": image}))
                if check["cosine_min"] < .999 or check["relative_l2"] > .03:
                    raise RuntimeError(f"HMR2 TensorRT validation failed: {check}")
                checks.append(check)
            manifest["validation"]["hmr2"] = dict(agreement=checks,
                eager=cuda_benchmark(lambda: visual({"img": image})),
                tensorrt=cuda_benchmark(lambda: engine({"img": image})))
        manifest["tensorrt"] = trt.__version__
        manifest["artifacts"]["hmr2.engine"] = sha256(engine_path)
        print("hmr2", manifest["validation"]["hmr2"], flush=True)
        del engine, visual
        gc.collect()
        torch.cuda.empty_cache()
    if args.detector_backend == "tensorrt":
        build_detector(args, root, directory, manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Validated cache:", directory, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", default=".local/rgb_bfm")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmr-backend", choices=("torch", "tensorrt"), default="tensorrt")
    parser.add_argument("--detector-backend", choices=("torch", "tensorrt"), default="tensorrt")
    parser.add_argument("--workspace-mb", type=int, default=2048)
    parser.add_argument("--temporal-precision", choices=("fp16", "fp32"), default="fp16")
    build(parser.parse_args())
