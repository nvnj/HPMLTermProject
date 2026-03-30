#!/usr/bin/env python3
"""
baseline_profile.py — Method 1: Baseline GPU Inference Profiling
================================================================
CS 5463 – Parallel Computing, Spring 2026
Naveen John | UTSA ScooterLab / CARE AI Lab

Description:
    Profiles the existing YOLOv12 + Depth Pro sequential inference pipeline
    on recorded e-scooter ride video. Measures per-frame latency, GPU
    utilization, memory consumption, and end-to-end throughput (FPS).

    Both models execute sequentially on a single CUDA stream (default stream)
    in standard PyTorch eager mode — no TensorRT, no multi-stream overlap.

Usage:
    python baseline_profile.py --video <path_to_video> \
                               --yolo-model yolov12s.pt \
                               --max-frames 300 \
                               --output timing_baseline.csv

Requirements:
    pip install ultralytics torch torchvision opencv-python-headless pandas
    # Depth Pro: clone https://github.com/apple/ml-depth-pro and install
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_VIDEO = "ride_video.mp4"
DEFAULT_YOLO_MODEL = "yolov12s.pt"
DEFAULT_MAX_FRAMES = 300          # number of frames to profile
DEFAULT_OUTPUT = "timing_baseline.csv"
DEFAULT_WARMUP = 10               # warmup frames (discard from stats)
DEFAULT_FRAME_SIZE = (1280, 720)  # input resolution (W, H)
DEFAULT_YOLO_IMGSZ = 640          # YOLOv12 inference resolution
DEFAULT_YOLO_CONF = 0.25          # confidence threshold
DEFAULT_YOLO_IOU = 0.45           # NMS IoU threshold


# ---------------------------------------------------------------------------
# Helper: CUDA synchronization timer
# ---------------------------------------------------------------------------
class CUDATimer:
    """
    Uses CUDA events for accurate GPU timing.
    torch.cuda.Event(enable_timing=True) avoids CPU-GPU sync overhead
    that time.perf_counter() would introduce.
    """
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        torch.cuda.synchronize()  # wait for GPU to finish
        return self.start_event.elapsed_time(self.end_event)  # milliseconds


# ---------------------------------------------------------------------------
# Helper: GPU memory snapshot
# ---------------------------------------------------------------------------
def gpu_memory_mb():
    """Returns (allocated_MB, reserved_MB) for the current CUDA device."""
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    resv = torch.cuda.memory_reserved() / (1024 ** 2)
    return alloc, resv


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_yolo(model_path: str, device: str):
    """
    Load YOLOv12 via the Ultralytics API.
    Returns a model handle that accepts numpy frames.
    """
    from ultralytics import YOLO
    model = YOLO(model_path)
    model.to(device)
    # Trigger a dummy forward pass to initialize CUDA context
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=DEFAULT_YOLO_IMGSZ, verbose=False)
    return model


def load_depth_pro(device: str):
    """
    Load Apple Depth Pro model and preprocessing transform.
    Assumes ml-depth-pro is installed (pip install -e .)
    Returns (model, transform).
    """
    try:
        import depth_pro
        model, transform = depth_pro.create_model_and_transforms(device=device)
        model.eval()
        return model, transform
    except ImportError:
        print("[WARN] depth_pro not installed. Using mock depth model for profiling.")
        return None, None


# ---------------------------------------------------------------------------
# Mock depth model (for environments without Depth Pro installed)
# ---------------------------------------------------------------------------
class MockDepthModel:
    """
    Simulates Depth Pro's compute cost with a ViT-sized forward pass.
    Used only when the real model is unavailable — timing won't be exact,
    but the profiling harness and I/O pattern are identical.
    """
    def __init__(self, device):
        self.device = device
        # Build a stand-in network with comparable FLOPs (~300M params)
        # Using a simple conv stack as placeholder
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 256, 7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1, 1),
        ).to(device).eval()

    @torch.no_grad()
    def infer(self, frame_tensor):
        return self.net(frame_tensor)


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------
def preprocess_for_depth(frame_bgr, transform, device):
    """
    Converts an OpenCV BGR frame to the tensor format Depth Pro expects.
    If no real transform is available (mock mode), does basic normalization.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if transform is not None:
        # Real Depth Pro transform expects a PIL image
        from PIL import Image
        pil_img = Image.fromarray(frame_rgb)
        tensor = transform(pil_img).unsqueeze(0).to(device)
    else:
        # Mock: simple resize + normalize
        resized = cv2.resize(frame_rgb, (512, 384))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)
    return tensor


# ---------------------------------------------------------------------------
# Main profiling loop
# ---------------------------------------------------------------------------
def profile_baseline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[WARN] No CUDA device found. Timings will reflect CPU execution.")

    print(f"Device       : {device}")
    if device == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Video        : {args.video}")
    print(f"YOLO model   : {args.yolo_model}")
    print(f"Max frames   : {args.max_frames}")
    print(f"Warmup frames: {args.warmup}")
    print(f"Output       : {args.output}")
    print("-" * 60)

    # --- Load models ---
    print("Loading YOLOv12...")
    yolo = load_yolo(args.yolo_model, device)
    print("Loading Depth Pro...")
    depth_model, depth_transform = load_depth_pro(device)
    use_mock = depth_model is None
    if use_mock:
        mock = MockDepthModel(device)
    print("Models loaded.\n")

    # --- Open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video info   : {width}x{height} @ {fps_video:.1f} FPS, {total_video_frames} total frames")

    # --- Timing storage ---
    timer_decode = CUDATimer() if device == "cuda" else None
    timer_yolo = CUDATimer() if device == "cuda" else None
    timer_depth = CUDATimer() if device == "cuda" else None
    timer_total = CUDATimer() if device == "cuda" else None

    records = []
    frame_idx = 0
    processed = 0

    print(f"\nProfiling {args.max_frames} frames (first {args.warmup} are warmup)...\n")

    while processed < args.max_frames:
        # ----- Frame decode (CPU-bound) -----
        t0_cpu = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video if short
            ret, frame = cap.read()
            if not ret:
                break
        t_decode_cpu = (time.perf_counter() - t0_cpu) * 1000  # ms

        if device == "cuda":
            timer_total.start()

        # ----- YOLOv12 Detection -----
        if device == "cuda":
            timer_yolo.start()
        results = yolo.predict(
            frame,
            imgsz=DEFAULT_YOLO_IMGSZ,
            conf=DEFAULT_YOLO_CONF,
            iou=DEFAULT_YOLO_IOU,
            verbose=False,
        )
        if device == "cuda":
            t_yolo = timer_yolo.stop()
        else:
            t_yolo = 0.0

        num_detections = len(results[0].boxes) if results else 0

        # ----- Depth Pro Estimation -----
        depth_input = preprocess_for_depth(frame, depth_transform, device)
        if device == "cuda":
            timer_depth.start()
        with torch.no_grad():
            if not use_mock:
                depth_map = depth_model.infer(depth_input)
            else:
                depth_map = mock.infer(depth_input)
        if device == "cuda":
            t_depth = timer_depth.stop()
        else:
            t_depth = 0.0

        if device == "cuda":
            t_total = timer_total.stop()
        else:
            t_total = t_decode_cpu + t_yolo + t_depth

        # ----- Memory snapshot -----
        if device == "cuda":
            mem_alloc, mem_resv = gpu_memory_mb()
        else:
            mem_alloc, mem_resv = 0.0, 0.0

        # ----- Record -----
        is_warmup = processed < args.warmup
        records.append({
            "frame_idx": frame_idx,
            "is_warmup": is_warmup,
            "decode_ms": round(t_decode_cpu, 3),
            "yolo_ms": round(t_yolo, 3),
            "depth_ms": round(t_depth, 3),
            "total_ms": round(t_total, 3),
            "num_detections": num_detections,
            "gpu_mem_alloc_mb": round(mem_alloc, 1),
            "gpu_mem_reserved_mb": round(mem_resv, 1),
        })

        if processed % 50 == 0:
            fps_inst = 1000.0 / t_total if t_total > 0 else 0
            tag = " (warmup)" if is_warmup else ""
            print(f"  Frame {processed:4d}/{args.max_frames}  |  "
                  f"YOLO {t_yolo:6.1f}ms  Depth {t_depth:6.1f}ms  "
                  f"Total {t_total:6.1f}ms  ({fps_inst:.1f} FPS)  "
                  f"Dets={num_detections}{tag}")

        frame_idx += 1
        processed += 1

    cap.release()

    # ----- Write CSV -----
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nTiming data saved to {args.output}")

    # ----- Summary statistics (exclude warmup) -----
    data = [r for r in records if not r["is_warmup"]]
    if not data:
        print("[WARN] No non-warmup frames recorded.")
        return

    yolo_times = [r["yolo_ms"] for r in data]
    depth_times = [r["depth_ms"] for r in data]
    total_times = [r["total_ms"] for r in data]

    print("\n" + "=" * 60)
    print("SUMMARY (excluding warmup frames)")
    print("=" * 60)
    print(f"  Frames profiled : {len(data)}")
    print(f"  YOLO   — mean: {np.mean(yolo_times):6.1f}ms  "
          f"std: {np.std(yolo_times):5.1f}ms  "
          f"min: {np.min(yolo_times):5.1f}ms  max: {np.max(yolo_times):5.1f}ms")
    print(f"  Depth  — mean: {np.mean(depth_times):6.1f}ms  "
          f"std: {np.std(depth_times):5.1f}ms  "
          f"min: {np.min(depth_times):5.1f}ms  max: {np.max(depth_times):5.1f}ms")
    print(f"  Total  — mean: {np.mean(total_times):6.1f}ms  "
          f"std: {np.std(total_times):5.1f}ms  "
          f"min: {np.min(total_times):5.1f}ms  max: {np.max(total_times):5.1f}ms")
    mean_fps = 1000.0 / np.mean(total_times)
    print(f"  Throughput      : {mean_fps:.2f} FPS")
    print(f"  GPU Mem (alloc) : {np.mean([r['gpu_mem_alloc_mb'] for r in data]):.0f} MB avg")
    print(f"  GPU Mem (resv)  : {np.mean([r['gpu_mem_reserved_mb'] for r in data]):.0f} MB avg")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Method 1: Baseline GPU Inference Profiling for YOLOv12 + Depth Pro"
    )
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO,
                        help="Path to input ride video (default: ride_video.mp4)")
    parser.add_argument("--yolo-model", type=str, default=DEFAULT_YOLO_MODEL,
                        help="YOLOv12 model weights (default: yolov12s.pt)")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help="Number of frames to profile (default: 300)")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help="Warmup frames to discard (default: 10)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output CSV for timing data (default: timing_baseline.csv)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile_baseline(args)
