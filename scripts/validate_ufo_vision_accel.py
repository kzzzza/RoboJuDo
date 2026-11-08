"""Compare accelerated features/u/z against FP32 on identical real image crops."""

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYAV_LOGGING", "off")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for variable, relative in (("XDG_CACHE_HOME", ".local/cache"), ("TORCH_HOME", ".local/cache/torch"),
                           ("HF_HOME", ".local/cache/huggingface"), ("CUDA_CACHE_PATH", ".local/cache/cuda"),
                           ("TMPDIR", ".local/setup"), ("MPLCONFIGDIR", ".local/video_runtime"),
                           ("YOLO_CONFIG_DIR", ".local/video_runtime")):
    (ROOT / relative).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(ROOT / relative))

import numpy as np
import torch

from robojudo.video.vision_accel import (CachedTemporal, HalfVisual, TensorRTVisual,
                                        agreement, trt_detector_predictor, validated_cache)
from robojudo.video.vision_worker import _load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--packages", default=str(ROOT / ".local/rgb_bfm"))
    parser.add_argument("--accel-dir", default=str(ROOT / ".local/vision_accel"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hmr-backend", choices=("torch", "tensorrt"), required=True)
    parser.add_argument("--start-frame", type=int, default=120)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    bundle = _load_bundle(Path(args.packages) / "original_pipeline")
    cache = validated_cache(args.accel_dir, bundle.ROOT, args.device)
    if not cache:
        raise RuntimeError("Build and validate the acceleration cache first")
    import cv2
    from ultralytics import YOLO
    from lib.util.hmr2_preprocess import get_bbx_xys_from_xyxy, prepare_crops

    detector = YOLO(str(bundle.ROOT / "models/yolov8x.pt"))
    accelerated_detector = None
    predictor = None
    if "yolov8x.engine" in cache["artifacts"]:
        accelerated_detector = YOLO(str(Path(args.accel_dir) / "yolov8x.engine"), task="detect")
        predictor = trt_detector_predictor(args.device)
    box_ious = []
    capture = cv2.VideoCapture(args.video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    crops = []
    try:
        while len(crops) < 50:
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError("Need 50 detected frames for numerical validation")
            result = detector.predict(bgr, device=args.device, conf=.5, classes=[0], verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            if not len(boxes):
                continue
            box = boxes[np.argmax(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1))]
            if accelerated_detector is not None:
                candidate = accelerated_detector.predict(bgr, predictor=predictor, device=args.device,
                                    half=True, conf=.5, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy()
                if not len(candidate):
                    raise RuntimeError("Accelerated detector lost a person found by the FP32 detector")
                candidate = candidate[np.argmax(np.prod(candidate[:, 2:] - candidate[:, :2], axis=1))]
                intersection = np.maximum(0., np.minimum(box[2:], candidate[2:]) - np.maximum(box[:2], candidate[:2])).prod()
                union = np.prod(box[2:] - box[:2]) + np.prod(candidate[2:] - candidate[:2]) - intersection
                box_ious.append(float(intersection / max(float(union), 1e-8)))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            half = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2))
            square = get_bbx_xys_from_xyxy(torch.from_numpy(box[None]), base_enlarge=1.2)
            crops.append(prepare_crops(half[None], square, image_scale=.5).to(args.device))
    finally:
        capture.release()
    del detector, accelerated_detector
    gc.collect()
    torch.cuda.empty_cache()

    with torch.inference_mode():
        visual = bundle.load_visual(args.device)
        reference_features = torch.cat([visual({"img": crop}) for crop in crops])
        del visual
        gc.collect()
        torch.cuda.empty_cache()
        if args.hmr_backend == "tensorrt":
            visual = TensorRTVisual(Path(args.accel_dir) / "hmr2.engine", args.device, cache["tensorrt"])
        else:
            visual = HalfVisual(bundle.load_visual(args.device))
        actual_features = torch.cat([visual({"img": crop}) for crop in crops])
        checks = {"hmr2": agreement(reference_features, actual_features)}
        del visual, crops
        gc.collect()
        torch.cuda.empty_cache()
        frontend = bundle.load_frontend(args.device)
        accelerated = CachedTemporal(args.accel_dir, args.device, cache["temporal_precision"])
        spec = importlib.util.spec_from_file_location("validation_uz_head",
                        Path(args.packages) / "tcn_pipeline/code/uz_head.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = torch.load(Path(args.packages) / "tcn_pipeline/models/distilled_head.pt",
                             map_location="cpu", weights_only=True)
        head = module.UZHead(**payload["model_config"]).to(args.device).eval()
        head.load_state_dict(payload["state_dict"], strict=True)
        for valid in (0, 1, 3, 8):
            mask = torch.zeros(1, 16, dtype=torch.bool, device=args.device)
            mask[:, :valid] = True
            context = torch.zeros(1, 16, 1024, device=args.device)
            actual_context = torch.zeros_like(context)
            context[:, :valid] = reference_features.mean(0)
            actual_context[:, :valid] = actual_features.mean(0)
            reference_u = frontend(reference_features[None], context_kv={"img_kv": context},
                                    window_mask=mask, return_motion_feature=True).reshape(1, 50, 1024)
            actual_u = accelerated(actual_features[None], actual_context, mask).reshape(1, 50, 1024)
            checks[f"tcam_{valid}_past"] = agreement(reference_u, actual_u)
            valid_frames = torch.ones(1, 50, dtype=torch.bool, device=args.device)
            reference_z = head(reference_u, valid_frames)
            actual_z = head(actual_u, valid_frames)
            checks[f"z_{valid}_past"] = agreement(reference_z, actual_z)
            if not torch.allclose(torch.linalg.vector_norm(actual_z, dim=-1),
                                  torch.full((1, 50), 16., device=args.device), atol=1e-3):
                raise RuntimeError("Accelerated z does not preserve norm 16")
    for stage, check in checks.items():
        if check["cosine_min"] < .999 or check["relative_l2"] > .03:
            raise RuntimeError(f"FP32 agreement failed at {stage}: {check}")
    report = dict(status="passed", robot_connected=False, frames=50,
                  hmr_backend=args.hmr_backend, checks=checks,
                  note="Identical crops and temporal inputs; cadence changes are validated separately")
    if box_ious:
        report["detector_box_iou_min"] = min(box_ious)
        report["detector_box_iou_mean"] = float(np.mean(box_ious))
        if min(box_ious) < .98:
            raise RuntimeError(f"Detector boxes differ from FP32: min IoU {min(box_ious):.6f}")
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
