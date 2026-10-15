# LLM Image & Video Classifier

A self-hosted, local-first web app and CLI for organizing and analyzing visual media with local AI models. Classify photos, search by semantic meaning, write tags into EXIF/IPTC/XMP metadata non-destructively, split long videos into scenes, and transcribe speech with Whisper.

---

## Index

- [Use Cases](#use-cases)
  - [1. Photo Organization & Metadata Tagging](#1-photo-organization--metadata-tagging)
  - [2. Video Scene Detection & Speech Transcription](#2-video-scene-detection--speech-transcription)
- [Quickstart](#quickstart)
  - [Prerequisite: Local Vision Model](#prerequisite-local-vision-model)
  - [1. Docker](#1-docker)
  - [2. Python Virtual Environment (Env)](#2-python-virtual-environment-env)
  - [3. Apple Containers (Podman / OrbStack)](#3-apple-containers-podman--orbstack)
  - [Apple `container` (no Docker)](#apple-container-no-docker)
  - [4. Mac / Desktop App (Electron)](#4-mac--desktop-app-electron)
  - [5. Command Line (CLI)](#5-command-line-cli)
- [How to Use Guide](#how-to-use-guide)
  - [Single Photo Analysis](#single-photo-analysis)
  - [Workflows (Chaining Models)](#workflows-chaining-models)
  - [Batch Processing & Similarity Search](#batch-processing--similarity-search)
  - [Metadata Export](#metadata-export)
  - [Video Scene Detection & Transcription](#video-scene-detection--transcription)
  - [Tagging Video Scenes](#tagging-video-scenes)
- [Suggested Models](#suggested-models)
  - [Recommended Vision LLMs](#recommended-vision-llms)
  - [Dedicated Vision Models](#dedicated-vision-models)
  - [Whisper Speech Models](#whisper-speech-models)
- [Documentation Index](#documentation-index)
- [License](#license)

---

## Use Cases

### 1. Photo Organization & Metadata Tagging

Use local Multimodal LLMs (like Qwen or Gemma) alongside fast computer vision models (SigLIP 2, EfficientNetV2, ResNet-50, RT-DETRv2) to understand your photo collection.

- **Non-Destructive Tagging:** Writes categories, captions, ratings, and tags directly into EXIF, IPTC, and XMP tags on file copies—leaving your originals untouched.
- **Semantic Search & Similarity:** Groups near-duplicates and visually similar photos automatically with SigLIP 2 embeddings, or search using plain English descriptions (`like:"kids playing in snow"`).
- **Batch Processing:** Run multi-model workflows over entire directories with progress tracking, pause/resume, and instant filtering.

![Single photo view: RT-DETRv2 detecting objects with EXIF data below and structured JSON on the right](docs/screenshot.jpg)

*Photo: [Retired electrician](https://commons.wikimedia.org/wiki/File:Moscow_2025-08-10_bicycle_race,_sidewalk_02.jpg), CC0, via Wikimedia Commons.*

---

### 2. Video Scene Detection & Speech Transcription

Process videos of any length without loading entire files into memory.

- **Scene Detection:** Detect hard cuts, dissolves, and fades using TransNetV2, PySceneDetect, or FFmpeg with instant real-time threshold tuning.
- **Audio Transcription:** Automatically transcribe spoken audio with timestamped subtitles (SRT, VTT, JSON) using faster-whisper.
- **Speaker Diarization:** Identify who speaks when with pyannote.audio, so transcript lines, subtitles, and the timeline show each speaker.
- **Video with Subtitles:** Download the original video with the transcript as a switchable subtitle track (MKV), or burned into the picture (MP4).
- **Scene Tagging & Export:** Classify scene keyframes using your LLM/vision workflows and download complete packages with thumbnails, speech transcripts, and CSV/JSON summaries.

![Video Classifier view: NASA video split into scenes with threshold curve, speech activity, and Whisper transcript](docs/screenshot-video.jpg)

*Video: [Why Did NASA Choose Asteroid Bennu?](https://svs.gsfc.nasa.gov/14398), NASA SVS, Public Domain via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Why_Did_NASA_Choose_Asteroid_Bennu-_(SVS14398).webm).*

---

## Quickstart

### Prerequisite: Local Vision Model
Start any OpenAI-compatible server with a vision model (e.g. via `llama.cpp` or `ollama`):

```sh
# Using llama-server:
llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF --host 0.0.0.0 --port 8080

# Or using Ollama:
ollama run qwen2.5-vl
```

---

### 1. Docker
Fastest multi-service setup (Web UI + Vision + Video):

**`.env` options for Docker:**
```env
# Required: LLM endpoint (host.docker.internal reaches the host on macOS/Windows)
LLM_BASE_URL=http://host.docker.internal:8080/v1

# Optional overrides:
# LLM_API_KEY=your-api-key          # Optional bearer token
# LLM_TIMEOUT=300                  # Request timeout in seconds
# VIDEO_DATA=/path/to/host/videos   # Host path to store videos (bypasses VM disk limits)
# VIDEO_RETENTION_DAYS=7           # Auto-delete video temp files after N days
# VISION_MAX_LOADED=2              # Max vision models in RAM at once
```

**Commands:**
```sh
echo 'LLM_BASE_URL=http://host.docker.internal:8080/v1' > .env
docker compose up --build -d
```
Open **<http://localhost:8090>** in your browser.

> On Linux, add `extra_hosts: ["host.docker.internal:host-gateway"]` to `compose.yaml` if connecting to host services.

---

### 2. Python Virtual Environment (Env)
Run natively on bare-metal Python without containers:

**`.env` options for Python:**
```env
# Required: local LLM server endpoint
LLM_BASE_URL=http://localhost:8080/v1

# Optional (if running auxiliary vision/video services locally):
# VISION_BASE_URL=http://localhost:8000
# VIDEO_BASE_URL=http://localhost:8001
# LLM_API_KEY=
# LLM_TIMEOUT=300
```

**Commands:**
```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

LLM_BASE_URL=http://localhost:8080/v1 uvicorn app.main:app --port 8090
```
Open **<http://localhost:8090>**.  
*(See [Local Environment Guide](docs/local-environment.md) to run auxiliary vision and video services locally).*

---

### 3. Apple Containers (Podman / OrbStack)
Optimized for Apple Silicon macOS using Podman or OrbStack:

**`.env` options for Apple Containers:**
```env
# Required: Podman routes host traffic through host.containers.internal
LLM_BASE_URL=http://host.containers.internal:8080/v1

# Recommended: store videos on host filesystem to avoid Podman VM disk limits
# VIDEO_DATA=/Volumes/Media/video-data
# VIDEO_RETENTION_DAYS=7
```

**Commands:**
```sh
echo 'LLM_BASE_URL=http://host.containers.internal:8080/v1' > .env
podman compose up --build -d
```
Open **<http://localhost:8090>**.

---

### Apple `container` (no Docker)
Apple's own [`container`](https://github.com/apple/container) runs the same `Containerfile` and `compose.yaml` without Docker Desktop:

```sh
brew install container && container system start
./scripts/apple-container.sh up
```
Open **<http://127.0.0.1:8090>**. The script reads the same `.env` and works out the Mac's address itself. See [Apple container guide](docs/apple-container.md) for the differences from Docker and speed numbers.

---

### 4. Mac / Desktop App (Electron)
Launch as a dedicated desktop window without browser URL bars:

**Options & Setup:**
- Connects to your local backend (`http://localhost:8090` from Docker or Python Env).
- Respects the backend `.env` configuration for models and storage paths.

**Commands:**
```sh
# Option A: Fast browser app mode (Chrome/Brave):
open -a "Google Chrome" --args --app="http://localhost:8090"

# Option B: Standalone macOS .app bundle using Nativefier / Electron:
npx -y nativefier --name "Photo Classifier" "http://localhost:8090"
```
*(See [Desktop & Electron Guide](docs/desktop-electron.md) for custom Electron wrappers).*

---

### 5. Command Line (CLI)
Batch process photos directly from your terminal using exported web UI settings:

**`.env` options for CLI:**
```env
# CLI reads .env automatically from the working directory:
LLM_BASE_URL=http://localhost:8080/v1
VISION_BASE_URL=http://127.0.0.1:8000
# LLM_API_KEY=your-api-key
```

**Commands:**
```sh
python3 -m venv .venv-cli && .venv-cli/bin/pip install -r requirements-cli.txt
docker compose up -d vision  # auxiliary vision service on port 8000

.venv-cli/bin/python -m app.cli catalog.settings.json ~/Pictures/Trip --tag --zip
```
*(See [CLI Guide](docs/cli.md) for full command-line arguments and semantic search flags).*

---

## How to Use Guide

### Single Photo Analysis
1. Drop a photo onto the dropzone (or press `Cmd/Ctrl+O`).
2. Select a model or workflow from the dropdown.
3. Review extracted EXIF data, prompt settings, and press **Run** (`Cmd/Ctrl+Enter`).
4. Output JSON is validated and displayed alongside raw model output.

### Workflows (Chaining Models)
Workflows execute a pipeline of vision and LLM tasks:
- **Variable Interpolation:** Reference earlier steps in downstream prompts, e.g. `{{objects.counts.person}}`, `{{scene.summary}}`, or `{{exif.DateTimeOriginal}}`.
- **Pre-configured Workflows:** Includes built-in workflows for structured cataloging, deep captioning, and object verification.

### Batch Processing & Similarity Search
- **Folder Ingestion:** Select a local directory to populate a thumbnail grid.
- **State & Resuming:** Color-coded borders show processing states (grey: pending, green: complete, red: failed). Interrupted batches resume automatically.
- **Smart Grouping:** Toggle **Group › Similar photos** to cluster visually similar images using SigLIP 2 embeddings. Adjust the **Alike** slider (e.g. `0.9` for bursts/duplicates, `0.8` for looser groupings).
- **Semantic Text Search:** Type `like:"kids at a birthday party"` to rank images by natural language relevance without re-running models.

### Metadata Export
- Click **Export image** or **Export all…** in Batch view.
- Tagged copies are produced with predictions written to `ImageDescription`, `Keywords`, `Subject`, `Rating`, `City`, `Event`, and `PersonInImage`.
- Originals are strictly read-only and never modified.  
*(See [Metadata Export Mapping](docs/metadata-export.md) for tag specifications).*

### Video Scene Detection & Transcription
1. Switch to **Video Classifier** in the top navigation and drop a video file.
2. Select a detector (**TransNetV2**, **PySceneDetect**, or **FFmpeg**) and click **Detect scenes**.
3. Drag the dashed threshold line on the timeline to interactively tune scene boundaries.
4. Open the **Transcript** tab, select a Whisper model, and click **Transcribe** to generate time-aligned subtitles.
5. Click **Identify speakers** to label who is talking (requires a one-time [Hugging Face token](docs/video-classifier.md#speaker-diarization-pyannote)). Choose **Auto speakers** or a fixed count; every transcript of that audio track then shows its speakers.
6. Click **Video** to download the video with subtitles, either as a subtitle track you can toggle in the player (instant, full quality) or burned into the picture at 360p or original size.

### Tagging Video Scenes
1. Switch to **Tag scenes** to inspect individual scene frames.
2. Click **Tag all** to run your chosen vision model or workflow across every scene.
3. Use `{{exif.Speech}}` in prompts to pass the transcribed spoken audio for each scene directly to the LLM.
4. Export a complete zip containing scene thumbnails, `scenes.json`, `scenes.csv`, and SRT subtitles.

---

## Suggested Models

### Recommended Vision LLMs
All recommended models are Apache-2.0, support multimodal vision input, and run directly in `llama.cpp` using `-hf` or in Ollama / LM Studio:

| Model | Download | Recommended For | `llama-server` Flag |
|---|---|---|---|
| **Gemma 4 E4B** | 5 GB | Fast tagging, captions, and categorization on laptops | `-hf ggml-org/gemma-4-E4B-it-GGUF` |
| **Qwen3-VL 8B Instruct** | 6 GB | Reading text in images (OCR, signs, documents); direct answers | `-hf Qwen/Qwen3-VL-8B-Instruct-GGUF` |
| **Qwen3.5 9B** | 7 GB | High-quality reasoning and rich scene descriptions | `-hf unsloth/Qwen3.5-9B-GGUF` |
| **Gemma 4 26B A4B** | 15 GB | Mixture-of-Experts architecture; fast inference with detailed output | `-hf ggml-org/gemma-4-26B-A4B-it-GGUF` |
| **Qwen3.6 35B A3B** | 21 GB | State-of-the-art vision understanding and complex scene reasoning | `-hf ggml-org/Qwen3.6-35B-A3B-GGUF` |

> **Tips:**
> - **Context Window:** Images at 1536 px require ~1,500 context tokens. Add `-c 8192` to your `llama-server` command.
> - **Thinking Models:** For models that think before answering, raise **Max tokens** to 4096 or start the server with `--reasoning off`.
> - **Ollama:** Use `http://host.docker.internal:11434/v1` as your base URL.

### Dedicated Vision Models
Lightweight CPU models hosted in the auxiliary `vision` service:

| Model | Function | Description |
|---|---|---|
| `siglip2-base` | Zero-shot & Embeddings | Scores custom label sets and generates 768-d vectors for similarity search |
| `efficientnetv2-s` | Classification | Fast ImageNet-1k classification |
| `resnet50` | Classification | Standard ImageNet-1k benchmark classifier |
| `rtdetr-v2-s` | Object Detection | Real-time object detection and bounding boxes across 80 COCO classes |

### Whisper Speech Models
CPU-optimized speech transcription in the `video` service via `faster-whisper`:

| Model | Size | Speed vs Accuracy |
|---|---|---|
| `tiny` / `base` | 78–145 MB | Fastest; best for quick previews or clear audio |
| `small` | 488 MB | **Recommended default:** Excellent balance of speed and precision |
| `medium` | 1.5 GB | Higher precision for complex terminology or accents |
| `large-v3 turbo` | 1.6 GB | Near large-v3 quality with significantly improved inference speed |
| `large-v3` | 3.1 GB | Maximum transcription fidelity |

### Speaker Diarization Models
Speaker identification in the `video` service via [pyannote.audio](https://github.com/pyannote/pyannote-audio) `speaker-diarization-3.1` (33 MB, CPU). The models are gated on Hugging Face: accept the terms of [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1) and [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0), then set `HF_TOKEN` in `.env`. The token is only needed for the first download.

---

## Documentation Index

Detailed guides are available in the [`docs/`](docs/) directory:

- [Configuration & Environment Variables](docs/configuration.md) — Port settings, volumes, retention policies, and tuning.
- [CLI Reference](docs/cli.md) — Command-line flags, automated batch runs, embeddings, and semantic queries.
- [Video Classifier & Scene Detection](docs/video-classifier.md) — Scene detectors, Whisper models, speaker identification, subtitled video, and video storage.
- [Metadata Export Mapping](docs/metadata-export.md) — EXIF, IPTC, and XMP tag mapping rules and safety protections.
- [Dedicated Vision Models](docs/vision-models.md) — Architecture, memory limits, and adding custom vision models.
- [Local Environment Setup](docs/local-environment.md) — Step-by-step instructions for running without containers.
- [Apple `container`](docs/apple-container.md) — Running under Apple's container runtime, how it differs from Docker, and speed.
- [Desktop App & Electron](docs/desktop-electron.md) — Packaging as a native desktop window on macOS.

---

## License

[MIT](LICENSE)

Model licences: the speaker-embedding weights used by pyannote (`wespeaker-voxceleb-resnet34-LM`) are [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), so attribution is required if you redistribute them; pyannote.audio, `segmentation-3.0`, and the 3.1 pipeline are MIT. Whisper and faster-whisper are MIT.
