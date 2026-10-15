# Dedicated Vision Models

In addition to multimodal Large Language Models, the application provides specialized, lightweight computer vision models running locally on CPU in the `vision` service.

## Built-in Vision Models

| Model Identifier | Type | Output / Labels | Source |
|---|---|---|---|
| `siglip2-base` | Zero-shot & Embeddings | Custom user-supplied tags or 768-d vector | Google SigLIP 2 |
| `efficientnetv2-s` | Classification | ImageNet-1k categories | `tf_efficientnetv2_s.in21k_ft_in1k` (timm) |
| `resnet50` | Classification | ImageNet-1k categories | `resnet50.a1_in1k` (timm) |
| `rtdetr-v2-s` | Real-Time Object Detection | COCO 80 classes with bounding boxes | `PekingU/rtdetr_v2_r18vd` |

## Key Features

- **On-Demand Caching**: Model weights download automatically upon first request into the persistent `vision-cache` volume.
- **LRU Memory Management**: The service limits simultaneously active models to `VISION_MAX_LOADED` (default: 2) to control RAM usage. Least-recently used models are evicted automatically.
- **Semantic Embeddings**: `siglip2-base` computes unified text-image embeddings (`POST /embed`), enabling visual similarity grouping (`similar>0.85`) and natural language semantic image search (`like:"query"`).
- **CPU Optimized**: Inference runs entirely on CPU without requiring CUDA, taking ~40ms to ~800ms per image depending on model complexity and host hardware.

## Adding Custom Vision Models

To register a new vision model:
1. Open `vision/server.py`.
2. Add a new `Spec` entry to the `SPECS` registry defining the model initialization, weights repo, preprocessor, and prediction wrapper.
3. Restart or rebuild the `vision` container.

