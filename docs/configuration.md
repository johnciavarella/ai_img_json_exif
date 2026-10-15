# Configuration & Environment Variables

This document details all configuration parameters and environment variables used by the LLM Image and Video Classifier.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://host.containers.internal:8080/v1` | Default OpenAI-compatible LLM endpoint. Editable in the UI. For Docker on macOS/Windows, use `http://host.docker.internal:8080/v1`. |
| `LLM_API_KEY` | *(empty)* | Optional Bearer token if your model server requires authentication. |
| `LLM_TIMEOUT` | `300` | Request timeout in seconds for LLM inference requests. |
| `VISION_BASE_URL` | `http://vision:8000` (or `http://localhost:8000`) | URL where the auxiliary vision service is reachable. |
| `VISION_MAX_LOADED` | `2` | Number of vision models kept in RAM simultaneously; least recently used (LRU) models are unloaded first. |
| `VIDEO_BASE_URL` | `http://video:8001` (or `http://localhost:8001`) | URL where the video processing service is reachable. |
| `VIDEO_DATA` | `video-data` (Docker volume) | Path or volume where uploaded videos are stored. Set to a host path (e.g. `/Volumes/Media/video-data`) to avoid container disk limits. |
| `HF_TOKEN` | *(empty)* | Hugging Face read token for the video service, needed once to download the gated pyannote speaker models (accept their terms first; see [Video Classifier](video-classifier.md#speaker-diarization-pyannote)). Without it, **Identify speakers** is disabled. |
| `VIDEO_RETENTION_DAYS` | `7` | Retention period in days before uploaded videos and their temporary artifacts are automatically cleaned up. |

## Persistence & Volumes

In Docker / Podman compose setups, the following volumes are used:

- `app-data`: Stores persistent application data such as saved workflows (`/data`).
- `vision-cache`: Caches downloaded model weights for SigLIP 2, EfficientNetV2, ResNet-50, and RT-DETRv2 (`/cache`).
- `video-cache`: Caches downloaded TransNetV2 weights, Whisper models, and pyannote speaker models (`/cache`).
- `video-data`: Stores raw uploaded videos and processed scene frames/previews (`/videos`).

> **Security Note:** The web application does not implement built-in authentication. Always host it behind a reverse proxy (e.g., Caddy, NGINX, Cloudflare Tunnel, or Tailscale) or run it on a strictly trusted local network.

