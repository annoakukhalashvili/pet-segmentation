"""Final evaluation of all models on the held-out TEST set + error analysis of the best model.

Run this ONCE, after all training decisions were made on the validation set.

Outputs in artifacts/results/:
    test_results.json / test_results.md     comparison table E0-E3
    error_analysis/per_image.csv            per-image metrics of the best model
    error_analysis/summary.json             per-species and per-breed averages
    error_analysis/worst.png, best.png      image | ground truth | prediction

Usage:
    python scripts/evaluate_models.py
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from petseg.data.annotations import NUM_CLASSES, parse_split_file
from petseg.data.dataset import PetSegmentationDataset
from petseg.data.splits import read_split
from petseg.data.transforms import build_transforms, denormalize
from petseg.env import select_device
from petseg.evaluation.metrics import SegmentationMetrics
from petseg.models.checkpoint import load_model
from petseg.models.deeplab import build_pretrained_voc_model, voc_logits_to_pet_labels
from petseg.visualization import overlay_mask

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS = {"e1_head_only": "E1 head only", "e2_full_finetune": "E2 full fine-tune", "e3_weighted_loss": "E3 weighted loss"}


@torch.no_grad()
def evaluate_with_per_image(predict_fn, loader, image_ids, device):
    """Return (overall metrics dict, list of per-image metric dicts)."""
    overall = SegmentationMetrics(NUM_CLASSES)
    per_image, index = [], 0
    for images, masks in tqdm(loader, desc="Evaluating", leave=False):
        predictions = predict_fn(images.to(device)).cpu()
        overall.update(predictions, masks)
        for prediction, mask in zip(predictions, masks):
            single = SegmentationMetrics(NUM_CLASSES)
            single.update(prediction, mask)
            per_image.append({"image_id": image_ids[index], "mean_iou": single.compute()["mean_iou"],
                              "foreground_iou": single.compute()["foreground_iou"]})
            index += 1
    return overall.compute(), per_image


def save_gallery(dataset, predict_fn, device, rows, title, out_path):
    """Each row: image | ground truth overlay | prediction overlay."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
    index_of = {image_id: i for i, image_id in enumerate(dataset.image_ids)}
    for r, row in enumerate(rows):
        image, mask = dataset[index_of[row["image_id"]]]
        with torch.no_grad():
            prediction = predict_fn(image.unsqueeze(0).to(device))[0].cpu().numpy()
        rgb = (denormalize(image).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        panels = [(rgb, row["image_id"]), (overlay_mask(rgb, mask.numpy()), "ground truth"),
                  (overlay_mask(rgb, prediction), f"prediction (mIoU {row['mean_iou']:.2f})")]
        for c, (panel, label) in enumerate(panels):
            axes[r, c].imshow(panel)
            axes[r, c].set_title(label, fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def markdown_table(rows: list[dict]) -> str:
    columns = [("model", "Model"), ("val_mean_iou", "Val mIoU"), ("mean_iou", "Test mIoU"),
               ("foreground_iou", "Test FG IoU"), ("iou_background", "Test IoU bg"), ("iou_pet", "Test IoU pet"),
               ("iou_boundary", "Test IoU boundary"), ("mean_dice", "Test mean Dice"), ("pixel_accuracy", "Test pixel acc.")]
    lines = ["| " + " | ".join(h for _, h in columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join("-" if row.get(k) is None else str(row.get(k)) for k, _ in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all models on the test set.")
    parser.add_argument("--data-dir", default="data/oxford-iiit-pet", type=Path)
    parser.add_argument("--split-file", default="splits/test.txt", type=Path)
    parser.add_argument("--experiments-dir", default="artifacts/experiments", type=Path)
    parser.add_argument("--out-dir", default="artifacts/results", type=Path)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gallery-size", default=6, type=int)
    args = parser.parse_args()

    device = select_device(args.device)
    image_ids = read_split(args.split_file)
    dataset = PetSegmentationDataset(args.data_dir, image_ids, build_transforms(train=False, image_size=320))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info("Test set: %d images, device=%s", len(dataset), device)

    rows, per_image_by_model, predictors = [], {}, {}

    voc_model = build_pretrained_voc_model().to(device).eval()
    predictors["e0"] = lambda x: voc_logits_to_pet_labels(voc_model(x)["out"])
    e0_val_file = args.out_dir / "e0_zero_shot_val.json"
    e0_val = json.loads(e0_val_file.read_text())["mean_iou"] if e0_val_file.is_file() else None
    metrics, per_image_by_model["e0"] = evaluate_with_per_image(predictors["e0"], loader, image_ids, device)
    rows.append({"model": "E0 zero-shot", "key": "e0", "val_mean_iou": e0_val, **metrics})
    logger.info("E0: test mIoU %s", metrics["mean_iou"])

    for key, label in EXPERIMENTS.items():
        checkpoint = args.experiments_dir / key / "best.pt"
        if not checkpoint.is_file():
            logger.warning("Skipping %s: %s not found", key, checkpoint)
            continue
        model, _ = load_model(checkpoint, device)
        predictors[key] = (lambda m: (lambda x: m(x)["out"].argmax(dim=1)))(model)
        val = json.loads((args.experiments_dir / key / "best_val_metrics.json").read_text())["val_mean_iou"]
        metrics, per_image_by_model[key] = evaluate_with_per_image(predictors[key], loader, image_ids, device)
        rows.append({"model": label, "key": key, "val_mean_iou": val, **metrics})
        logger.info("%s: test mIoU %s", label, metrics["mean_iou"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "test_results.json").write_text(json.dumps(rows, indent=2))
    table = markdown_table(rows)
    (args.out_dir / "test_results.md").write_text(table + "\n")
    print("\n" + table + "\n")

    # ----- Error analysis of the best model, chosen by VALIDATION mean IoU -----
    trained = [r for r in rows if r["key"] != "e0"]
    best = max(trained, key=lambda r: r["val_mean_iou"])
    logger.info("Best model (by validation mIoU): %s", best["model"])
    per_image = per_image_by_model[best["key"]]

    records = {r.image_id: r for r in parse_split_file(args.data_dir / "annotations" / "test.txt")}
    for row in per_image:
        row["species"] = records[row["image_id"]].species
        row["breed"] = records[row["image_id"]].breed

    analysis_dir = args.out_dir / "error_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis_dir / "per_image.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    def group_mean(field):
        groups = defaultdict(list)
        for row in per_image:
            groups[row[field]].append(row["mean_iou"])
        return {k: round(float(np.mean(v)), 4) for k, v in sorted(groups.items(), key=lambda kv: np.mean(kv[1]))}

    by_breed = group_mean("breed")
    scores = np.array([r["mean_iou"] for r in per_image])
    summary = {
        "model": best["model"],
        "per_image_mean_iou": {"mean": round(float(scores.mean()), 4), "median": round(float(np.median(scores)), 4),
                               "share_below_0.5": round(float((scores < 0.5).mean()), 4)},
        "by_species": group_mean("species"),
        "worst_5_breeds": dict(list(by_breed.items())[:5]),
        "best_5_breeds": dict(list(by_breed.items())[-5:]),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Error analysis summary: %s", json.dumps(summary, indent=2))

    ranked = sorted(per_image, key=lambda r: r["mean_iou"])
    save_gallery(dataset, predictors[best["key"]], device, ranked[: args.gallery_size],
                 f"{best['model']}: worst test predictions", analysis_dir / "worst.png")
    save_gallery(dataset, predictors[best["key"]], device, ranked[-args.gallery_size:],
                 f"{best['model']}: best test predictions", analysis_dir / "best.png")
    logger.info("Saved results to %s", args.out_dir)


if __name__ == "__main__":
    main()
