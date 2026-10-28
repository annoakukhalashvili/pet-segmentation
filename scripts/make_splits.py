"""Create reproducible train / val / test splits without cross-split duplicates.

- test  = the official Oxford-IIIT Pet test list (never used for training or tuning)
- train / val = the official trainval list, split per breed (stratified)
- unusable samples (corrupted, mismatched, invalid masks) are excluded
- near-duplicate images across splits are removed to prevent leakage:
    train copy of a train-val or train-test pair -> removed from train
    val copy of a val-test pair                  -> removed from val
  The official test set is never changed, so results stay comparable.

Outputs (small text files, committed to Git so everyone uses identical splits):
    splits/train.txt, splits/val.txt, splits/test.txt
    splits/excluded.json     every removed image and the reason
    splits/duplicates.json   near-duplicate pairs found (before removal)

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --keep-duplicates   # report only, remove nothing
"""

import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from petseg.data.annotations import parse_split_file
from petseg.data.duplicates import dhash, find_near_duplicates
from petseg.data.splits import stratified_split, write_split
from petseg.data.validate import check_sample

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# (split that loses its copy, other split). Test always keeps its copy.
DUPLICATE_CHECKS = (("train", "val"), ("train", "test"), ("val", "test"))


def usable_records(base_dir: Path, split: str, excluded: dict) -> list:
    """Return the records of an official split whose files pass validation."""
    records = parse_split_file(base_dir / "annotations" / f"{split}.txt")
    usable = []
    for record in tqdm(records, desc=f"Validating {split}"):
        report = check_sample(
            record.image_id,
            base_dir / "images" / f"{record.image_id}.jpg",
            base_dir / "annotations" / "trimaps" / f"{record.image_id}.png",
        )
        if report.is_usable:
            usable.append(record)
        else:
            excluded[record.image_id] = {"reason": "unusable", "official_split": split, "errors": report.errors}
    return usable


def find_cross_split_duplicates(base_dir: Path, splits: dict, max_distance: int) -> dict:
    """Return {"train-val": [(id_a, id_b, distance), ...], ...} for every pair of splits."""
    hashes = {}
    for ids in splits.values():
        for image_id in tqdm(ids, desc="Hashing", leave=False):
            hashes[image_id] = dhash(base_dir / "images" / f"{image_id}.jpg")

    duplicates = {}
    for name_a, name_b in DUPLICATE_CHECKS:
        ids_a, ids_b = splits[name_a], splits[name_b]
        duplicates[f"{name_a}-{name_b}"] = find_near_duplicates(
            ids_a, [hashes[i] for i in ids_a], ids_b, [hashes[i] for i in ids_b], max_distance
        )
    return duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test splits.")
    parser.add_argument("--data-dir", default="data/oxford-iiit-pet", type=Path)
    parser.add_argument("--out-dir", default="splits", type=Path)
    parser.add_argument("--val-fraction", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-hash-distance", default=4, type=int)
    parser.add_argument("--keep-duplicates", action="store_true", help="only report duplicates")
    args = parser.parse_args()

    if not (args.data_dir / "annotations").is_dir():
        raise SystemExit(f"Dataset not found in {args.data_dir}. Run scripts/download_data.py first.")

    excluded: dict = {}
    trainval = usable_records(args.data_dir, "trainval", excluded)
    test = usable_records(args.data_dir, "test", excluded)

    train_ids, val_ids = stratified_split(trainval, args.val_fraction, args.seed)
    splits = {"train": train_ids, "val": val_ids, "test": sorted(r.image_id for r in test)}

    duplicates = find_cross_split_duplicates(args.data_dir, splits, args.max_hash_distance)
    for key, pairs in duplicates.items():
        logger.info("near-duplicate pairs %s: %d", key, len(pairs))

    if not args.keep_duplicates:
        for name_a, name_b in DUPLICATE_CHECKS:
            for id_a, id_b, distance in duplicates[f"{name_a}-{name_b}"]:
                if id_a in excluded:  # already removed because of another pair
                    continue
                excluded[id_a] = {"reason": "near-duplicate", "removed_from": name_a, "duplicate_of": id_b, "distance": distance}
        removed = set(excluded)
        splits = {name: [i for i in ids if i not in removed] for name, ids in splits.items()}

    # Sanity checks: the splits must not share any image, and test must be unchanged.
    assert not set(splits["train"]) & set(splits["val"]), "train and val overlap"
    assert not (set(splits["train"]) | set(splits["val"])) & set(splits["test"]), "trainval and test overlap"
    assert len(splits["test"]) == len(test), "the official test set must not change"

    for name, ids in splits.items():
        write_split(args.out_dir / f"{name}.txt", ids)
        logger.info("%-5s %d images", name, len(ids))
    logger.info("excluded %d images in total", len(excluded))

    with open(args.out_dir / "excluded.json", "w", encoding="utf-8") as file:
        json.dump(excluded, file, indent=2)
    with open(args.out_dir / "duplicates.json", "w", encoding="utf-8") as file:
        json.dump({k: [{"a": a, "b": b, "distance": d} for a, b, d in v] for k, v in duplicates.items()}, file, indent=2)
    logger.info("Saved splits to %s", args.out_dir)


if __name__ == "__main__":
    main()
