# Experiment Report

## Setup common to all trained models

| Setting | Value |
|---|---|
| Model | DeepLabV3-MobileNetV3-Large, torchvision `COCO_WITH_VOC_LABELS_V1` weights, final layer → 3 classes |
| Data | train 3,105 / val 549 / test 3,669 (breed-stratified, duplicates removed) |
| Input | 320x320; train: RandomResizedCrop (scale 0.5–1.0) + horizontal flip; eval: resize |
| Loss | Cross-entropy (E3: median-frequency class weights) |
| Optimizer | AdamW, weight decay 1e-4; head LR 1e-3, backbone LR 1e-4; cosine schedule |
| Budget | 10 epochs, early stopping patience 5, batch size 16, seed 42 |
| Hardware | Kaggle NVIDIA T4, mixed precision; about 21 s/epoch (E1), about 26 s/epoch (E2, E3) |
| Selection | Best epoch by validation mean IoU |

## Results

### Validation (used for all decisions)

| Model | Best epoch | mIoU | Foreground IoU | IoU bg | IoU pet | IoU boundary | Mean Dice |
|---|---|---|---|---|---|---|---|
| E0 zero-shot | – | 0.5499 | 0.8385 | 0.8964 | 0.7533 | 0.0000 | 0.6016 |
| E1 head only | 10 / 10 | 0.7770 | 0.9028 | 0.9318 | 0.8676 | 0.5315 | 0.8626 |
| E2 full fine-tune | 10 / 10 | 0.7880 | 0.9079 | 0.9354 | 0.8745 | 0.5543 | 0.8710 |
| E3 weighted loss | 9 / 10 | 0.7594 | 0.8747 | 0.9049 | 0.8549 | 0.5185 | 0.8516 |

### Test (evaluated once, after selection)

See [`results/test_results.md`](results/test_results.md). E2: test mIoU 0.7958, foreground IoU 0.9132, boundary IoU 0.5781.

Per-epoch training curves: `results/e*_history.csv`.

## Interpretation

- **E0 → E1:** training only the head lifts mIoU from 0.55 to 0.78. Most of the gain comes from learning the
  boundary class, but foreground IoU also improves (0.84 → 0.90), because the head adapts to this dataset's
  annotation style.
- **E1 → E2:** unfreezing the backbone adds about 0.011 mIoU (val) and 0.012 (test), consistent on both
  splits. A small effect from a single seed, so treat it as suggestive.
- **E2 → E3:** class weighting lowered every metric, including the boundary IoU it targeted. The hypothesis is rejected.
- **Undertraining:** E1 and E2 peaked at the final epoch, so a longer schedule would likely improve them.

## Optimization experiment

See [`results/benchmark.md`](results/benchmark.md). Measured on CPU (Apple M1, 4 threads), batch size 1,
model forward pass only, median of 50 runs after 10 warm-up runs. Quality was measured on the validation
set against the same 320 px ground truth.

- ONNX Runtime vs PyTorch at 320 px: **3.05x faster** (103.3 → 33.9 ms), with identical mIoU (0.788).
- 224 px input: 1.45x faster in PyTorch and 4.25x with ONNX Runtime, but mIoU falls by 0.048 and boundary IoU by 0.076.
- Recommended deployment: ONNX Runtime at 320 px.

## Error analysis (E2, test set)

Figures: [`figures/worst.png`](figures/worst.png), [`figures/best.png`](figures/best.png). Summary: [`results/error_analysis_summary.json`](results/error_analysis_summary.json).

| Statistic | Value |
|---|---|
| Per-image mIoU, mean / median | 0.785 / 0.800 |
| Images below 0.5 mIoU | 0.84% |
| Cats / dogs | 0.789 / 0.783 |
| Worst breeds | japanese_chin 0.756, miniature_pinscher 0.760, german_shorthaired 0.762 |
| Best breeds | newfoundland 0.813, Bombay 0.809, leonberger 0.806 |

Worst six images: three close-ups whose ground truth is entirely background, and two with inverted masks.
In all five, the prediction is visually correct, so these are **annotation errors**. The remaining one is a
real failure: a cat's beige bed segmented as part of the cat.
