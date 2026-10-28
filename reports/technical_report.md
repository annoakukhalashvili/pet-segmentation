# Technical Report: Three-Class Pet Segmentation

## 1. What problem does the project solve?

The system takes a photo of a cat or dog and labels every pixel as **background**, **pet** or
**boundary** (the uncertain outline of the animal). Pixel-level masks support tasks such as
background removal, measuring an animal's position and size in a frame, and speeding up the
labelling of new data. The output is shown in a Gradio application that accepts any new image.

## 2. Why were this dataset, model and approach selected?

**Dataset: Oxford-IIIT Pet.** Compared with the other suggested datasets, it offered the best balance:
- Enough images for reliable validation and test sets (7,349; Penn-Fudan has only 170).
- Ready-made segmentation masks with a simple label scheme (Pascal VOC has 21 classes, and BDD100K is too large for free compute).
- A natural extra challenge: the three-class trimap includes an ambiguous boundary class.

**Model: DeepLabV3 with a MobileNetV3-Large backbone (torchvision).**
- Small (44 MB) and fast on CPU, so the demo runs on a laptop without a GPU.
- Its published weights already recognise the Pascal VOC classes `cat` and `dog`. That allowed a real
  zero-shot baseline (E0) and made transfer learning efficient.
- A stable, well-documented PyTorch implementation, so there was no need for extra frameworks.

**Approach: transfer learning.** Training from scratch would need far more data and compute. We
reuse the pretrained network, replace only the final 1x1 convolution (21 → 3 classes) and fine-tune.

## 3. Experiments and results

All experiments share: 320x320 inputs, batch size 16, AdamW (weight decay 1e-4), cosine learning-rate
schedule, a 10-epoch budget with early stopping (patience 5), seed 42, and mixed precision on a
Kaggle T4 GPU. The best epoch was selected by **validation mean IoU**. The test set was used once, after all decisions.

| ID | Change (one variable at a time) | Hypothesis | Val mIoU | Test mIoU | Test boundary IoU | Outcome |
|---|---|---|---|---|---|---|
| E0 | Pretrained model, no training (`cat`/`dog` → pet) | Decent foreground, no boundary | 0.550 | 0.545 | 0.000 | Baseline |
| E1 | Train the head only (backbone frozen) | Beats E0 and learns the boundary class | 0.777 | 0.784 | 0.557 | Confirmed |
| E2 | Also train the backbone (vs E1) | Higher mIoU than E1 | 0.788 | 0.796 | 0.578 | Confirmed (small gain) |
| E3 | Add median-frequency class weights (vs E2) | Higher boundary IoU | 0.759 | 0.775 | 0.559 | **Rejected** |

- **Fine-tuning works.** Test foreground IoU rose from 0.829 (E0) to 0.913 (E2), and the model learned
  the boundary class (0 → 0.578 IoU).
- **Full fine-tuning helps a little.** E2 beats E1 by about 0.012 mIoU on both validation and test.
  This is a single run per configuration, so part of the gap may be seed variation.
- **The weighted loss made every metric worse**, including boundary IoU, which it was meant to improve.
  Our explanation (not proven): the weights (about 0.5 / 1.0 / 2.7 for background / pet / boundary) push
  the model to over-predict boundary pixels, and the boundary labels are ambiguous, so the extra
  weight amplifies label noise instead of signal.
- Validation and test scores agree closely and rank the models identically, so model selection did not overfit the validation set.

**Error analysis (E2, test set).**
- The mean per-image mIoU is 0.785 and the median 0.800. Only 0.84% of images (about 31) score below 0.5.
- Performance is even across species (cats 0.789, dogs 0.783) and breeds (worst: japanese_chin 0.756; best: newfoundland 0.813).
- Inspecting the six worst images showed that **at least four are annotation errors**. Three close-ups have
  an empty mask (the whole image is labelled background), and two have inverted masks (the background
  is labelled as pet). In these, the model's prediction is visually correct.
- One clear model error: a cat lying upside down in a beige bed, where the bed was segmented as part of the
  pet (similar colour and texture, unusual pose).
- The error categories are therefore: (1) label noise, (2) object-like background with similar texture,
  (3) thin structures and edges, where the boundary class is ambiguous by nature.

**Optimization (CPU, Apple M1, 4 threads, batch 1, validation set).** A 2x2 design separates the two effects:

| Configuration | Val mIoU | Median latency | Speed-up |
|---|---|---|---|
| PyTorch 320 px | 0.788 | 103.3 ms | 1.00x |
| ONNX Runtime 320 px | 0.788 | 33.9 ms | 3.05x |
| PyTorch 224 px | 0.740 | 71.0 ms | 1.45x |
| ONNX Runtime 224 px | 0.740 | 24.3 ms | 4.25x |

ONNX Runtime gives a **3.05x speed-up with identical accuracy**: its graph optimizations fuse operations,
and it avoids Python overhead. Lowering the resolution to 224 px adds speed but costs 0.048 mIoU and
0.076 boundary IoU. That is a trade-off, not a free improvement. Model size is unchanged (about 44 MB),
because neither change reduces the number of parameters.

## 4. Difficulties and how they were resolved

- **Data leakage risk.** A perceptual-hash check found 27 near-duplicate pairs across splits (for example,
  the same photo in colour and in grayscale). The train/val copies were removed and the official test set
  was left unchanged, so results stay comparable.
- **Frozen backbone still changing.** Freezing weights does not stop BatchNorm layers from updating
  their running statistics in training mode. The backbone is therefore also put in eval mode, and a
  test verifies it stays unchanged.
- **A `.gitignore` bug.** The pattern `data/` also excluded the source package `src/petseg/data/`, so
  training on Kaggle failed with `ModuleNotFoundError`. It was fixed by anchoring the patterns (`/data/`).
- **Fair baseline comparison.** The pretrained model cannot predict the boundary class, so a
  foreground IoU metric (pet and boundary merged) was added to compare all models fairly.
- **Limited compute and time.** Training ran on free Kaggle GPUs with checkpointing and resume support;
  the epoch budget was reduced to 10 to meet the deadline.
- **Local resource limits.** The laptop ran out of disk space during testing; it was resolved by removing
  caches and leftover archives.

## 5. Strengths and limitations

**Strengths.**
- A complete, reproducible pipeline: validated data, committed splits, pinned versions, saved configs, 53 automated tests.
- Controlled experiments that change one variable at a time, including an honestly reported negative result (E3).
- A strict validation/test separation.
- A measured optimization with a clear trade-off analysis.
- A working demo on CPU.

**Limitations.**
- Only 10 epochs. E1 and E2 peaked in the final epoch, so they are probably undertrained.
- A single seed per configuration, so there are no confidence intervals.
- Metrics are computed on 320x320 resized masks rather than at the original resolution.
- Label noise in the dataset caps the achievable score, especially for the boundary class.
- No public deployment.
- The ONNX export uses PyTorch's deprecated TorchScript exporter.

## 6. What would be improved with more time or compute?

1. Train longer (20–50 epochs) and repeat each configuration with three or more seeds, to report means and standard deviations.
2. Evaluate at the original image resolution.
3. Try boundary-aware losses (Dice, Lovász-Softmax) instead of class weighting, and a higher input resolution for thin structures.
4. Detect and correct noisy labels: the per-image scores already flag candidate errors.
5. Compare a stronger backbone (for example, ResNet-50 DeepLabV3 or SegFormer) with the speed/accuracy trade-off.
6. INT8 quantization in ONNX Runtime for further CPU speed-ups, with accuracy re-measured.
7. Deploy the demo publicly (for example, on Hugging Face Spaces) using the ONNX model.
