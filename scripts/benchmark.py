"""Optimization experiment: PyTorch vs ONNX Runtime, at 320 and 224 px input, on CPU.

2 x 2 design so each effect is measured separately:
    runtime:    PyTorch eager  vs  ONNX Runtime (graph optimizations)
    resolution: 320 px         vs  224 px
Quality is measured on the VALIDATION set, always against the same 320 px ground truth
(predictions from 224 px inputs are upsampled back to 320 px before scoring).
Latency: batch size 1, model forward only, median over many runs after warm-up, fixed thread count.

Usage:
    python scripts/benchmark.py
"""

import argparse
import json
import logging
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from petseg.data.annotations import NUM_CLASSES
from petseg.data.dataset import PetSegmentationDataset
from petseg.data.splits import read_split
from petseg.data.transforms import build_transforms
from petseg.evaluation.metrics import SegmentationMetrics
from petseg.inference.onnx_export import create_session, export_onnx
from petseg.models.checkpoint import load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EVAL_SIZE = 320


def make_runner(backend, model, onnx_path, threads):
    """Return a function: (1, 3, S, S) float tensor -> (1, 3, S, S) logits tensor."""
    if backend == "pytorch":
        def run(x):
            with torch.no_grad():
                return model(x)["out"]
        return run
    session = create_session(onnx_path, threads)
    return lambda x: torch.from_numpy(session.run(None, {"image": x.numpy()})[0])


def measure_latency(run, size, warmup, runs):
    x = torch.randn(1, 3, size, size)
    for _ in range(warmup):
        run(x)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        run(x)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return round(statistics.median(times), 1), round(times[int(0.9 * len(times)) - 1], 1)


def measure_quality(run, dataset, size, max_images):
    metrics = SegmentationMetrics(NUM_CLASSES)
    for i in tqdm(range(min(len(dataset), max_images)), desc=f"Quality @{size}", leave=False):
        image, mask = dataset[i]  # image at 320 px
        x = image.unsqueeze(0)
        if size != EVAL_SIZE:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        logits = run(x)
        if size != EVAL_SIZE:
            logits = F.interpolate(logits, size=(EVAL_SIZE, EVAL_SIZE), mode="bilinear", align_corners=False)
        metrics.update(logits.argmax(dim=1)[0], mask)
    return metrics.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference optimizations on CPU.")
    parser.add_argument("--weights", default="artifacts/weights/petseg_deeplabv3_mbv3.pt", type=Path)
    parser.add_argument("--data-dir", default="data/oxford-iiit-pet", type=Path)
    parser.add_argument("--split-file", default="splits/val.txt", type=Path)
    parser.add_argument("--sizes", default=[320, 224], type=int, nargs="+")
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--runs", default=50, type=int)
    parser.add_argument("--max-images", default=10_000, type=int)
    parser.add_argument("--out-dir", default="artifacts/results/optimization", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)  # same thread budget for both runtimes
    model, _ = load_model(args.weights, "cpu")
    dataset = PetSegmentationDataset(args.data_dir, read_split(args.split_file), build_transforms(False, EVAL_SIZE))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "model.pt"
        torch.save(model.state_dict(), state_path)
        pytorch_mb = state_path.stat().st_size / 1e6

    rows = []
    for size in args.sizes:
        onnx_path = export_onnx(model, args.out_dir / f"model_{size}.onnx", size)
        sizes_mb = {"pytorch": pytorch_mb, "onnxruntime": onnx_path.stat().st_size / 1e6}
        for backend in ("pytorch", "onnxruntime"):
            run = make_runner(backend, model, onnx_path, args.threads)
            median_ms, p90_ms = measure_latency(run, size, args.warmup, args.runs)
            quality = measure_quality(run, dataset, size, args.max_images)
            row = {"config": f"{backend}@{size}", "backend": backend, "input_size": size,
                   "median_latency_ms": median_ms, "p90_latency_ms": p90_ms, "model_mb": round(sizes_mb[backend], 1),
                   "val_mean_iou": quality["mean_iou"], "val_foreground_iou": quality["foreground_iou"],
                   "val_iou_boundary": quality["iou_boundary"]}
            rows.append(row)
            logger.info("%s", row)

    baseline = next(r for r in rows if r["config"] == f"pytorch@{args.sizes[0]}")
    for row in rows:
        row["speedup_vs_baseline"] = round(baseline["median_latency_ms"] / row["median_latency_ms"], 2)

    (args.out_dir / "benchmark.json").write_text(json.dumps(
        {"threads": args.threads, "runs": args.runs, "warmup": args.warmup, "torch": torch.__version__, "results": rows},
        indent=2))
    header = ["config", "val_mean_iou", "val_foreground_iou", "val_iou_boundary", "median_latency_ms",
              "p90_latency_ms", "speedup_vs_baseline", "model_mb"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(r[h]) for h in header) + " |" for r in rows]
    table = "\n".join(lines)
    (args.out_dir / "benchmark.md").write_text(table + "\n")
    print("\n" + table + "\n")


if __name__ == "__main__":
    main()
