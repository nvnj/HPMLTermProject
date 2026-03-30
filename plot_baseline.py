#!/usr/bin/env python3
"""
plot_baseline.py — Generate performance plots for Method 1
==========================================================
Reads timing_baseline.csv and produces:
  1. Per-frame latency breakdown (stacked bar)
  2. Throughput (FPS) over frames
  3. GPU memory usage over frames
  4. Latency distribution histogram

Usage:
    python plot_baseline.py --input timing_baseline.csv --outdir plots/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Exclude warmup frames
    df = df[df["is_warmup"] == False].reset_index(drop=True)
    return df


def plot_latency_breakdown(df, outdir):
    """Stacked bar chart: decode vs YOLO vs Depth per frame."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    ax.bar(x, df["decode_ms"], label="Frame Decode (CPU)", color="#4CAF50", alpha=0.8)
    ax.bar(x, df["yolo_ms"], bottom=df["decode_ms"], label="YOLOv12 Detection", color="#2196F3", alpha=0.8)
    ax.bar(x, df["depth_ms"], bottom=df["decode_ms"] + df["yolo_ms"],
           label="Depth Pro Estimation", color="#FF9800", alpha=0.8)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Frame Latency Breakdown — Baseline Sequential Pipeline")
    ax.legend(loc="upper right")
    ax.set_xlim(-1, len(df))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "latency_breakdown.png"), dpi=150)
    plt.close()
    print("  Saved latency_breakdown.png")


def plot_throughput(df, outdir):
    """Rolling-window FPS over time."""
    fps = 1000.0 / df["total_ms"]
    rolling = fps.rolling(window=20, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fps.values, alpha=0.3, color="#2196F3", label="Instantaneous FPS")
    ax.plot(rolling.values, color="#0D47A1", linewidth=2, label="Rolling Avg (20 frames)")
    ax.axhline(fps.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {fps.mean():.1f} FPS")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Throughput (FPS)")
    ax.set_title("Inference Throughput — Baseline Sequential Pipeline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "throughput_fps.png"), dpi=150)
    plt.close()
    print("  Saved throughput_fps.png")


def plot_gpu_memory(df, outdir):
    """GPU memory usage over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["gpu_mem_alloc_mb"], label="Allocated", color="#E91E63", linewidth=1.5)
    ax.plot(df["gpu_mem_reserved_mb"], label="Reserved (Cache)", color="#9C27B0", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("GPU Memory (MB)")
    ax.set_title("GPU Memory Usage — Baseline Sequential Pipeline")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "gpu_memory.png"), dpi=150)
    plt.close()
    print("  Saved gpu_memory.png")


def plot_latency_histogram(df, outdir):
    """Histogram of per-component latencies."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, label, color in [
        (axes[0], "yolo_ms", "YOLOv12", "#2196F3"),
        (axes[1], "depth_ms", "Depth Pro", "#FF9800"),
        (axes[2], "total_ms", "Total (end-to-end)", "#4CAF50"),
    ]:
        data = df[col]
        ax.hist(data, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(data.mean(), color="red", linestyle="--", label=f"Mean={data.mean():.1f}ms")
        ax.axvline(data.median(), color="black", linestyle=":", label=f"P50={data.median():.1f}ms")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=8)
    plt.suptitle("Latency Distributions — Baseline Sequential Pipeline", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "latency_histogram.png"), dpi=150)
    plt.close()
    print("  Saved latency_histogram.png")


def plot_speedup_placeholder(outdir):
    """
    Placeholder speedup chart comparing batch sizes on single GPU.
    Since Method 1 is single-stream baseline, we show batch=1 throughput
    as the reference point. Methods 2 and 3 will add their data points.
    """
    batch_sizes = [1]
    fps_values = [5.2]  # placeholder — update with real data
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(["Baseline\n(Batch=1)"], fps_values, color="#2196F3", width=0.4)
    ax.set_ylabel("Throughput (FPS)")
    ax.set_title("Throughput Comparison (to be updated in Methods 2 & 3)")
    ax.set_ylim(0, max(fps_values) * 2)
    for i, v in enumerate(fps_values):
        ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "speedup_comparison.png"), dpi=150)
    plt.close()
    print("  Saved speedup_comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="timing_baseline.csv")
    parser.add_argument("--outdir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.input)
    print(f"Loaded {len(df)} frames from {args.input}\n")

    plot_latency_breakdown(df, args.outdir)
    plot_throughput(df, args.outdir)
    plot_gpu_memory(df, args.outdir)
    plot_latency_histogram(df, args.outdir)
    plot_speedup_placeholder(args.outdir)

    print(f"\nAll plots saved to {args.outdir}/")


if __name__ == "__main__":
    main()
