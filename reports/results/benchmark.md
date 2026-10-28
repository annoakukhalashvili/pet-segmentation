| config | val_mean_iou | val_foreground_iou | val_iou_boundary | median_latency_ms | p90_latency_ms | speedup_vs_baseline | model_mb |
|---|---|---|---|---|---|---|---|
| pytorch@320 | 0.788 | 0.9079 | 0.5543 | 103.3 | 107.5 | 1.0 | 44.3 |
| onnxruntime@320 | 0.788 | 0.9079 | 0.5543 | 33.9 | 34.4 | 3.05 | 44.1 |
| pytorch@224 | 0.7402 | 0.8787 | 0.4779 | 71.0 | 73.1 | 1.45 | 44.3 |
| onnxruntime@224 | 0.7402 | 0.8787 | 0.4779 | 24.3 | 25.6 | 4.25 | 44.1 |
