# Local Environment (Native Python venv)

If you prefer to run services natively without Docker containers or container runtimes, follow this guide.

## Prerequisites
- Python 3.10+
- `ffmpeg` (required for video processing: `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux)
- `exiftool` (required for non-destructive metadata writing: `brew install exiftool` or `sudo apt install libimage-exiftool-perl`)

---

## 1. Web Application Service

Create a virtual environment and launch the FastAPI / Uvicorn server:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run web service on port 8090 pointing to local LLM server
LLM_BASE_URL=http://localhost:8080/v1 uvicorn app.main:app --reload --port 8090
```

---

## 2. Vision Service (Optional, for Vision Models & Embeddings)

To enable SigLIP 2, EfficientNetV2, ResNet-50, and RT-DETRv2 locally:

```sh
python3 -m venv .venv-vision
source .venv-vision/bin/activate
pip install -r vision/requirements.txt

CACHE_DIR=./data/cache uvicorn vision.server:app --port 8000
```
Then configure the web app with `VISION_BASE_URL=http://localhost:8000`.

---

## 3. Video Service (Optional, for Scene Detection, Whisper & Speakers)

To enable TransNetV2, PySceneDetect, Whisper transcription, and speaker identification natively. ffmpeg must be installed, with libass for burning in subtitles (`brew install ffmpeg` has it). `HF_TOKEN` is needed once for the speaker models:

```sh
python3 -m venv .venv-video
source .venv-video/bin/activate
# torch first, so pyannote.audio's dependencies don't replace it (constraints.txt pins the versions)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -r video/requirements.txt -c video/constraints.txt

cd video
HF_TOKEN=hf_... VIDEO_DIR=../data/videos VIDEO_CACHE=../data/cache uvicorn server:app --port 8001
```
Then configure the web app with `VIDEO_BASE_URL=http://localhost:8001`.

