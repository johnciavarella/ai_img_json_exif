"""Video scene service: probes uploads, builds a low-res proxy, runs scene detectors, transcribes speech
and tells its speakers apart.

The web app writes uploads into VIDEO_DIR (a volume shared with this service); everything
derived from them lives next to the source in the same folder:

    <id>/meta.json        name, size, created (written by the app)
    <id>/source.part      while uploading; renamed to source<.ext> once complete
    <id>/info.json        ffprobe summary
    <id>/proxy.mp4        360p H.264 copy every detector reads, and the browser plays
    <id>/scenes/<key>.json  one result per detector + settings
    <id>/scores/<detector>.f32  per-frame scores of live detectors, to re-threshold instantly
    <id>/thumbs/<ms>.jpg  scene thumbnails for the page (120px), made on first request
    <id>/thumbs/<ms>_full.jpg  360p thumbnails for exports
    <id>/tags/<key>/<config>/  a result's scenes tagged by one model or workflow: <n>.json is
                          scene n's JSON, <n>.record.json the whole run that made it
    <id>/transcripts/<key>.json  speech as timed segments and words, one per model + language
    <id>/speakers/a<track>.json  who speaks when on an audio track, shared by its transcripts
    <id>/subtitled/<transcript key>-<360|full>.mp4  the video with a transcript burned in
    <id>/tmp/             scratch for a running job (16 kHz audio while transcribing or
                          diarizing, the subtitles being muxed or burned in)

Heavy work (proxy, detection, transcription, diarization, burning in subtitles) runs one job at a time on a worker thread; the UI polls status.
Tagging runs in the browser, scene by scene, like the image side's batch runs; this service only
supplies each scene's frame and keeps what comes back.
The video itself is never cut: results export as a scene list with thumbnails and tags.
"""

import csv
import hashlib
import io
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

import numpy as np

import diarize
import transcribe as speech
from detectors import DETECTORS, Cancelled, Context, curve_from_scores, min_frames

VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", "/videos"))
PROXY_HEIGHT = int(os.environ.get("VIDEO_PROXY_HEIGHT", "360"))
FRAME_EDGE = int(os.environ.get("VIDEO_FRAME_EDGE", "1536"))  # frames sent to models, as for photos
RETENTION_DAYS = float(os.environ.get("VIDEO_RETENTION_DAYS", "7"))
UPLOAD_RETENTION_DAYS = 1  # abandoned partial uploads
MAX_PROXY_FPS = 60
ID_RE = re.compile(r"^[0-9a-f]{32}$")
KEY_RE = re.compile(r"^[a-z0-9-]{1,64}$")

if threads := os.environ.get("VIDEO_THREADS"):
    import torch

    torch.set_num_threads(int(threads))
THREADS = int(os.environ.get("VIDEO_THREADS") or os.cpu_count() or 4)


# ---------- folders ----------

def folder(video_id: str) -> Path:
    path = VIDEO_DIR / video_id
    if not ID_RE.match(video_id) or not (path / "meta.json").is_file():
        raise HTTPException(404, "Unknown or expired video.")
    return path


def source_path(root: Path) -> Optional[Path]:
    return next((p for p in root.glob("source.*") if p.suffix != ".part"), None)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    tmp.replace(path)


def touch(root: Path) -> None:
    """Mark a video as in use, so retention counts from the last time it was opened."""
    meta = root / "meta.json"
    try:
        if time.time() - meta.stat().st_mtime > 3600:
            meta.touch()
    except OSError:
        pass


# ---------- jobs ----------

class Job:
    def __init__(self, video_id: str, kind: str, label: str, fn, key: Optional[str] = None):
        self.id = uuid.uuid4().hex
        self.video_id, self.kind, self.label, self.fn, self.key = video_id, kind, label, fn, key
        self.state = "queued"  # queued | running | done | error | cancelled
        self.progress = 0.0
        self.started: Optional[float] = None
        self.error: Optional[str] = None
        self.stage = ""  # what a job is doing before its progress starts, e.g. downloading a model
        self.cancel = threading.Event()

    def report(self, fraction: float) -> None:
        if self.cancel.is_set():
            raise Cancelled()
        self.progress = max(0.0, min(1.0, fraction))

    def public(self) -> dict:
        eta = None
        if self.state == "running" and self.started and self.progress > 0.02 and not self.stage:
            elapsed = time.time() - self.started
            eta = round(elapsed * (1 - self.progress) / self.progress)
        return {
            "id": self.id, "kind": self.kind, "label": self.label, "key": self.key, "state": self.state,
            "progress": round(self.progress, 4), "eta": eta, "error": self.error, "stage": self.stage,
        }


_queue: "queue.Queue[Job]" = queue.Queue()
_jobs: dict = {}  # video id -> {(kind, key): Job}, the latest job of each kind and key
_jobs_lock = threading.Lock()


def _latest(video_id: str, kind: str, key: Optional[str] = None) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(video_id, {}).get((kind, key))


def submit(video_id: str, kind: str, label: str, fn, key: Optional[str] = None) -> Job:
    with _jobs_lock:
        jobs = _jobs.setdefault(video_id, {})
        old = jobs.get((kind, key))
        if old and old.state in ("queued", "running"):
            return old
        job = jobs[(kind, key)] = Job(video_id, kind, label, fn, key)
    _queue.put(job)
    return job


def active_jobs(video_id: str) -> list:
    with _jobs_lock:
        return [j for j in _jobs.get(video_id, {}).values() if j.state in ("queued", "running")]


def cancel_jobs(video_id: str) -> int:
    jobs = active_jobs(video_id)
    for job in jobs:
        job.cancel.set()
    return len(jobs)


def worker() -> None:
    while True:
        job = _queue.get()
        if job.cancel.is_set():
            job.state = "cancelled"
            continue
        job.state, job.started = "running", time.time()
        try:
            job.fn(job)
            job.state, job.progress = "done", 1.0
        except Cancelled:
            job.state = "cancelled"
        except Exception as e:  # noqa: BLE001 - shown in the UI
            job.state, job.error = "error", str(e) or type(e).__name__


# ---------- ffmpeg ----------

def run_ffmpeg(args: list, job: Optional[Job] = None, duration: float = 0.0) -> None:
    """Run ffmpeg to completion, reporting progress to job from its -progress output."""
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", "-progress", "pipe:1", "-nostats", *args],
            stdout=subprocess.PIPE, stderr=err, text=True,
        )
        try:
            for line in proc.stdout:
                if job and duration and line.startswith("out_time_us="):
                    value = line.split("=", 1)[1].strip()
                    if value.isdigit():
                        job.report(min(0.999, int(value) / 1e6 / duration))
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        code = proc.wait()
        if code != 0:
            err.seek(0)
            raise RuntimeError(f"ffmpeg exited {code}: {err.read().decode(errors='replace').strip()[-800:]}")


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe could not read the file: {out.stderr.strip()[-500:]}")
    data = json.loads(out.stdout)
    streams = data.get("streams", [])
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    if not video:
        raise RuntimeError("The file has no video stream.")
    fmt = data.get("format", {})

    def rate(value) -> float:
        try:
            num, _, den = str(value).partition("/")
            return float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            return 0.0

    fps_text = video.get("avg_frame_rate") if rate(video.get("avg_frame_rate")) > 0 else video.get("r_frame_rate")
    fps = rate(fps_text)
    if not 1 <= fps <= 1000:
        fps_text, fps = "25", 25.0
    duration = float(fmt.get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise RuntimeError("Could not tell how long the video is.")
    rotation = 0
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(side["rotation"])
    width, height = video.get("width"), video.get("height")
    if rotation % 180:
        width, height = height, width
    tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}
    return {
        "duration": duration,
        "size": int(fmt.get("size") or path.stat().st_size),
        "format": fmt.get("format_long_name") or fmt.get("format_name"),
        "bit_rate": int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None,
        "created": tags.get("creation_time"),
        "video": {
            "index": video["index"], "codec": video.get("codec_name"), "profile": video.get("profile"),
            # fps unrounded: frame n is at n / fps, and 29.97 vs 30000/1001 drifts over hours
            "width": width, "height": height, "fps": fps, "fps_text": fps_text,
            "pix_fmt": video.get("pix_fmt"), "rotation": rotation,
        },
        "audio": [
            {"index": s["index"], "codec": s.get("codec_name"), "channels": s.get("channels"),
             "language": s.get("tags", {}).get("language")}
            for s in streams if s.get("codec_type") == "audio"
        ],
        "streams": len(streams),
    }


def prepare(job: Job) -> None:
    """Probe the source and build the proxy."""
    root = VIDEO_DIR / job.video_id
    src = source_path(root)
    if not src:
        raise RuntimeError("The upload is not complete.")
    info = probe(src)
    write_json(root / "info.json", info)
    job.report(0.01)

    fps_text = info["video"]["fps_text"] if info["video"]["fps"] <= MAX_PROXY_FPS else str(MAX_PROXY_FPS)
    gop = max(1, round(min(info["video"]["fps"], MAX_PROXY_FPS)))
    part = root / "proxy.part.mp4"
    run_ffmpeg([
        "-i", str(src),
        "-map", f"0:{info['video']['index']}", "-map", "0:a:0?",
        # Constant frame rate, so frame n of the proxy is at n / fps for every detector
        "-vf", f"scale=-2:'min({PROXY_HEIGHT},trunc(ih/2)*2)':flags=fast_bilinear,format=yuv420p",
        "-fps_mode", "cfr", "-r", fps_text,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-g", str(gop),
        "-c:a", "aac", "-b:a", "64k", "-ac", "2",
        "-movflags", "+faststart", "-f", "mp4", str(part),
    ], job, info["duration"])
    part.replace(root / "proxy.mp4")


def ensure_prepared(video_id: str, root: Path) -> Optional[Job]:
    """Queue the proxy for a complete upload that doesn't have one, unless that already failed."""
    if (root / "proxy.mp4").exists() or not source_path(root):
        return None
    job = _latest(video_id, "prepare")
    if job and job.state in ("error", "cancelled"):
        return job
    return submit(video_id, "prepare", "Preparing video", prepare)


# ---------- scenes ----------

_summaries: dict = {}  # path -> (mtime, summary), so status polls don't re-read every result


def scene_result(path: Path) -> dict:
    data = read_json(path)
    if data is None:
        raise HTTPException(404, "No such result.")
    return data


def summary(path: Path) -> Optional[dict]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _summaries.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    data = read_json(path)
    if not data:
        return None
    s = {k: data.get(k) for k in ("key", "detector", "label", "params", "created", "elapsed")}
    s["count"] = len(data.get("scenes", []))
    s["cuts"] = [sc["start"] for sc in data.get("scenes", [])[1:]]
    _summaries[path] = (mtime, s)
    return s


def save_result(root: Path, detector, params: dict, key: str, info: dict, cuts: list, frames: int,
                curve: dict, elapsed: float) -> None:
    fps = proxy_fps(info)
    cuts = sorted(c for c in cuts if 0 < c / fps < info["duration"])
    starts = [0.0] + [round(c / fps, 3) for c in cuts]
    ends = starts[1:] + [round(info["duration"], 3)]
    scenes = [
        {"start": s, "end": e, "start_frame": round(s * fps), "end_frame": round(e * fps)}
        for s, e in zip(starts, ends)
    ]
    (root / "scenes").mkdir(exist_ok=True)
    write_json(root / "scenes" / f"{key}.json", {
        "key": key, "detector": detector.id, "label": detector.label, "params": params,
        "created": time.time(), "elapsed": round(elapsed, 1), "live": detector.live,
        "fps": fps, "duration": info["duration"], "frames": frames, "scenes": scenes, "curve": curve,
    })


def proxy_fps(info: dict) -> float:
    return min(info["video"]["fps"], MAX_PROXY_FPS)


def scores_path(root: Path, detector_id: str) -> Path:
    return root / "scores" / f"{detector_id}.f32"


def rescore(root: Path, detector, params: dict, key: str, info: dict, scores: np.ndarray, elapsed: float) -> None:
    """Cut a live detector's saved scores at the given threshold."""
    threshold = params[detector.curve_param]
    cuts = detector.rescore(scores, threshold, min_frames(params, proxy_fps(info)))
    save_result(root, detector, params, key, info, cuts, len(scores), curve_from_scores(scores, threshold), elapsed)


def detect(detector_id: str, params: dict):
    detector = DETECTORS[detector_id]

    def run(job: Job) -> None:
        root = VIDEO_DIR / job.video_id
        info = read_json(root / "info.json")
        if not (root / "proxy.mp4").exists() or not info:
            raise RuntimeError("The video isn't ready yet.")
        started = time.time()
        ctx = Context(proxy=str(root / "proxy.mp4"), fps=proxy_fps(info), duration=info["duration"],
                      params=params, report=job.report)
        out = detector.run(ctx)
        if "scores" in out:
            path = scores_path(root, detector.id)
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_suffix(".tmp")
            out["scores"].astype("<f4").tofile(tmp)
            tmp.replace(path)
            rescore(root, detector, params, job.key, info, out["scores"], time.time() - started)
        else:
            save_result(root, detector, params, job.key, info, out["cuts"], out["frames"], out["curve"],
                        time.time() - started)

    return run


# ---------- transcripts ----------

def transcript_path(root: Path, key: str) -> Path:
    return root / "transcripts" / f"{key}.json"


def transcript_summary(path: Path) -> Optional[dict]:
    """What status needs: the settings, counts and where speech is, without the words."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _summaries.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    data = read_json(path)
    if not data:
        return None
    s = {k: data.get(k) for k in ("key", "model", "label", "params", "language", "created", "elapsed")}
    s["count"] = len(data["segments"])
    s["words"] = sum(len(seg.get("words") or []) for seg in data["segments"])
    s["spans"] = speech.spans(data["segments"])
    _summaries[path] = (mtime, s)
    return s


def extract_audio(job: Job, root: Path, info: dict, track: int, weight: float) -> Path:
    """Decode an audio track to 16 kHz mono s16le PCM in tmp/, for the first weight of the
    progress bar. The caller deletes it."""
    tmp = root / "tmp"
    tmp.mkdir(exist_ok=True)
    pcm = tmp / f"audio-{job.id}.pcm"
    # The proxy carries the first audio track and is far smaller than the source, which
    # ffmpeg would have to read end to end; other tracks come from the source
    proxy = root / "proxy.mp4"
    src, stream = ((proxy, "0:a:0") if track == 0 and proxy.exists()
                   else (source_path(root), f"0:{info['audio'][track]['index']}"))
    job.stage = "Extracting the audio"
    decoding = SimpleNamespace(report=lambda f: job.report(weight * f))
    try:
        run_ffmpeg(["-i", str(src), "-map", stream, "-vn", "-ac", "1", "-ar", str(speech.SAMPLE_RATE),
                    "-f", "s16le", str(pcm)], decoding, info["duration"])
    except BaseException:
        pcm.unlink(missing_ok=True)
        raise
    return pcm


def audio_info(root: Path) -> dict:
    info = read_json(root / "info.json")
    if not info:
        raise RuntimeError("The video isn't ready yet.")
    if not info["audio"]:
        raise RuntimeError("This video has no audio track.")
    return info


def run_transcribe(params: dict):
    def run(job: Job) -> None:
        root = VIDEO_DIR / job.video_id
        info = audio_info(root)
        started = time.time()
        weight = 0.03  # share of the progress bar for decoding
        pcm = extract_audio(job, root, info, params["track"], weight)
        try:
            out = speech.transcribe(
                pcm, params, lambda f: job.report(weight + (1 - weight) * f),
                lambda text: setattr(job, "stage", text), THREADS)
        finally:
            pcm.unlink(missing_ok=True)
        path = transcript_path(root, job.key)
        path.parent.mkdir(exist_ok=True)
        write_json(path, {
            "key": job.key, "engine": "faster-whisper", "model": params["model"], "label": speech.label(params),
            "params": params, "language": out["language"], "language_probability": out["language_probability"],
            "created": time.time(), "elapsed": round(time.time() - started, 1), "duration": info["duration"],
            "segments": out["segments"],
        })

    return run


def transcript_keys(root: Path, track: Optional[int] = None) -> list:
    """Keys of the video's transcripts, or of those of one audio track."""
    out = []
    for p in (root / "transcripts").glob("*.json"):
        s = transcript_summary(p)
        if s and (track is None or (s.get("params") or {}).get("track", 0) == track):
            out.append(p.stem)
    return out


# ---------- speakers ----------

def speakers_path(root: Path, track: int) -> Path:
    return root / "speakers" / f"a{track}.json"


def speakers_summary(path: Path) -> Optional[dict]:
    """What status needs: the settings, the count and each speaker's stretches for the timeline."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _summaries.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    data = read_json(path)
    if not data:
        return None
    s = {k: data.get(k) for k in ("params", "count", "created", "elapsed", "label")}
    s["track"] = data["params"]["track"]
    s["bands"] = diarize.bands(data["turns"])
    _summaries[path] = (mtime, s)
    return s


def run_diarize(params: dict):
    def run(job: Job) -> None:
        root = VIDEO_DIR / job.video_id
        info = audio_info(root)
        started = time.time()
        weight = 0.03
        pcm = extract_audio(job, root, info, params["track"], weight)
        try:
            out = diarize.diarize(pcm, params, lambda f: job.report(weight + (1 - weight) * f),
                                  lambda text: setattr(job, "stage", text))
        finally:
            pcm.unlink(missing_ok=True)
        path = speakers_path(root, params["track"])
        path.parent.mkdir(exist_ok=True)
        write_json(path, {
            "engine": "pyannote.audio", "model": diarize.MODEL, "label": diarize.LABEL,
            "params": params, "created": time.time(), "elapsed": round(time.time() - started, 1),
            "duration": info["duration"], "count": out["count"], "turns": out["turns"],
        })
        # Videos burned in with the old speakers (or none) are out of date
        drop_subtitled(root, transcript_keys(root, params["track"]))

    return run


def with_speakers(root: Path, data: dict) -> dict:
    """A transcript with its track's speakers, if diarized: lines split where the speaker
    changes and each carrying "speaker"."""
    found = read_json(speakers_path(root, (data.get("params") or {}).get("track", 0)))
    if not found:
        return data
    return {**data, "segments": diarize.assign(data["segments"], found["turns"]),
            "speakers": found["count"], "speakers_created": found["created"]}


# ---------- video with subtitles ----------
# Muxed as a subtitle track (stream copy, streamed as it's made) or burned into the picture (a
# job that re-encodes, saved under subtitled/)

BURN_SIZES = ("360", "full")


def subtitled_path(root: Path, key: str, size: str) -> Path:
    return root / "subtitled" / f"{key}-{size}.mp4"


def subtitled_list(root: Path) -> list:
    out = []
    for p in sorted((root / "subtitled").glob("*.mp4")):
        if p.name.endswith(".part.mp4"):
            continue
        key, _, size = p.stem.rpartition("-")
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"name": p.stem, "key": key, "size": size, "bytes": st.st_size, "created": st.st_mtime})
    return out


def drop_subtitled(root: Path, keys: list) -> None:
    for key in keys:
        for size in BURN_SIZES:
            subtitled_path(root, key, size).unlink(missing_ok=True)


def write_srt(root: Path, data: dict) -> Path:
    tmp = root / "tmp"
    tmp.mkdir(exist_ok=True)
    path = tmp / f"subs-{uuid.uuid4().hex}.srt"
    path.write_text(speech.to_srt(data["segments"]), encoding="utf-8")
    return path


def run_burn(key: str, size: str):
    def run(job: Job) -> None:
        root = VIDEO_DIR / job.video_id
        info = audio_info(root)
        data = with_speakers(root, transcript_result(root, key))
        track = (data.get("params") or {}).get("track", 0)
        srt = write_srt(root, data)
        out = subtitled_path(root, key, size)
        out.parent.mkdir(exist_ok=True)
        part = out.with_name(out.stem + ".part.mp4")
        # libass sizes SRT text against a 288-line canvas, so the same style suits both sizes;
        # fontconfig falls back to Noto CJK for what DejaVu doesn't cover
        style = "FontName=DejaVu Sans,FontSize=16,Outline=1.2,Shadow=0.6,MarginV=14"
        burn = f"subtitles=filename={srt}:force_style='{style}',format=yuv420p"
        if size == "360":
            # The proxy, with its own audio (the first track) or the transcript's track from the source
            inputs = ["-i", str(root / "proxy.mp4")]
            if track:
                inputs += ["-i", str(source_path(root))]
            maps = ["-map", "0:v:0", "-map", f"1:{info['audio'][track]['index']}" if track else "0:a:0?"]
            codecs = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                      *(["-c:a", "aac", "-b:a", "128k"] if track else ["-c:a", "copy"])]
        else:
            inputs = ["-i", str(source_path(root))]
            maps = ["-map", f"0:{info['video']['index']}", "-map", f"0:{info['audio'][track]['index']}"]
            codecs = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "192k"]
        try:
            run_ffmpeg([*inputs, *maps, "-vf", burn, *codecs, "-movflags", "+faststart", "-f", "mp4", str(part)],
                       job, info["duration"])
            part.replace(out)
        finally:
            srt.unlink(missing_ok=True)
            part.unlink(missing_ok=True)

    return run


async def mux_stream(proc: subprocess.Popen, first: bytes, cleanup):
    """Yield ffmpeg's output as it comes; stop ffmpeg if the download stops part-way. Async, so
    the download stopping cancels it and runs the finally: Starlette never closes a plain
    generator, which would leave ffmpeg blocked on a full pipe."""
    try:
        yield first
        while chunk := await run_in_threadpool(proc.stdout.read, 1 << 20):
            yield chunk
        await run_in_threadpool(proc.wait)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        cleanup()


# ---------- tags ----------
# A result's scenes tagged by a model or workflow, keyed like the image side's batch results
# ("workflow:<id>", "llm:<model>", ...). That key can hold any character, so its folder is named
# by a hash and config.json says which it is. One file per scene: a long run never rewrites what
# it already saved, and listing reads the small entries without the full run records.

def check_key(key: str) -> str:
    if not KEY_RE.match(key):
        raise HTTPException(400, "Invalid result key.")
    return key


def tags_dir(root: Path, key: str, config: str) -> Path:
    return root / "tags" / key / hashlib.sha1(config.encode()).hexdigest()[:16]


def tag_entries(path: Path) -> dict:
    """Scene index -> saved entry, without the run records."""
    out = {}
    for p in path.glob("*.json"):
        if p.stem.isdigit() and (entry := read_json(p)) is not None:
            out[int(p.stem)] = entry
    return out


def tag_configs(root: Path, key: str) -> list:
    """Every model or workflow that has tagged this result, most recent first."""
    out = []
    for p in (root / "tags" / key).glob("*/config.json"):
        cfg = read_json(p)
        if cfg:
            tagged = sum(1 for x in p.parent.glob("*.json") if x.stem.isdigit())
            out.append({**cfg, "tagged": tagged, "updated": p.parent.stat().st_mtime})
    return sorted(out, key=lambda c: c["updated"], reverse=True)


# ---------- export ----------
# A scene list describes the video without cutting it: each scene's times, frames and a
# thumbnail, ready to carry tags later. JSON, CSV and the zip are all built here so they match.

def timecode(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def thumb_time(scene: dict) -> float:
    """A little into the scene, past any transition (same as sceneRow in video.js)."""
    return round(scene["start"] + min((scene["end"] - scene["start"]) / 3, 2), 3)


def make_thumb(proxy: Path, ms: int, path: Path, height: Optional[int]) -> bool:
    """Write the proxy's frame at ms to path as a JPEG, unless it's already there. False if there
    is no frame at that time (past the end)."""
    if path.exists():
        return True
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex[:8]}.jpg")
    scale = ["-vf", f"scale=-2:{height}"] if height else []
    try:
        run_ffmpeg(["-ss", f"{ms / 1000:.3f}", "-i", str(proxy), "-frames:v", "1", *scale,
                    "-q:v", "5" if height else "3", str(tmp)])
    except RuntimeError:
        tmp.unlink(missing_ok=True)
        raise
    if not tmp.exists():
        return False
    tmp.replace(path)
    return True


def export_doc(root: Path, info: dict, result: dict, thumbnails: bool, config: str = "",
               transcript: Optional[dict] = None) -> dict:
    """The scene list; with config, each scene tagged by that model or workflow carries its JSON
    as "metadata"; with a transcript, each scene carries the words spoken in it as "speech"."""
    meta = read_json(root / "meta.json", {})
    v = info["video"]
    fps = v["fps"]  # frame numbers count frames of the original, not the proxy
    tags = tag_entries(tags_dir(root, result["key"], config)) if config else {}
    spoken = speech.speech_by_scene(transcript["segments"], result["scenes"]) if transcript else []
    scenes = []
    for i, sc in enumerate(result["scenes"], 1):
        row = {
            "scene": i, "start": sc["start"], "end": sc["end"], "duration": round(sc["end"] - sc["start"], 3),
            "start_timecode": timecode(sc["start"]), "end_timecode": timecode(sc["end"]),
            "start_frame": round(sc["start"] * fps), "end_frame": round(sc["end"] * fps),
            "thumbnail_time": thumb_time(sc),
        }
        if thumbnails:
            row["thumbnail"] = f"thumbnails/scene-{i:04d}.jpg"
        if transcript:
            row["speech"] = spoken[i - 1]
        tag = tags.get(i - 1)
        if tag and tag.get("ok"):
            row["metadata"] = tag["result"]
        scenes.append(row)
    tagged = sum(1 for row in scenes if "metadata" in row)
    tagging = None
    if tagged:
        label = (read_json(tags_dir(root, result["key"], config) / "config.json") or {}).get("label")
        tagging = {"config": config, "source": label or config, "tagged": tagged}
    return {
        "format": "photo-organizer/video-scenes", "version": 1,
        "exported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video": {
            "name": meta.get("name"), "size": info["size"], "duration": info["duration"],
            "fps": round(fps, 3), "width": v["width"], "height": v["height"], "codec": v["codec"],
            "container": info["format"], "recorded": info.get("created"),
        },
        "detector": {"id": result["detector"], "label": result["label"], "params": result["params"]},
        **({"tagging": tagging} if tagging else {}),
        **({"transcript": {"model": transcript["label"], "language": transcript["language"]}} if transcript else {}),
        "scene_count": len(scenes),
        "scenes": scenes,
    }


CSV_FIELDS = ["file", "scene", "start", "end", "duration", "start_timecode", "end_timecode",
              "start_frame", "end_frame", "thumbnail_time", "thumbnail", "speech", "metadata"]


def export_csv(doc: dict) -> str:
    buf = io.StringIO()
    # Optional columns only when some scene has them
    fields = [f for f in CSV_FIELDS if f not in ("thumbnail", "speech", "metadata") or any(f in r for r in doc["scenes"])]
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for row in doc["scenes"]:
        out = {"file": doc["video"]["name"], **row}
        if "metadata" in row:
            out["metadata"] = json.dumps(row["metadata"], ensure_ascii=False, separators=(",", ":"))
        w.writerow(out)
    return buf.getvalue()


class _Sink(io.RawIOBase):
    """Write-only buffer the zip writes into; the stream drains it as it goes (as in app/runs.py)."""

    def __init__(self):
        self.chunks = []

    def writable(self):
        return True

    def write(self, b):
        self.chunks.append(bytes(b))
        return len(b)

    def drain(self):
        chunks, self.chunks = self.chunks, []
        return chunks


def export_zip(root: Path, doc: dict, folder_name: str, transcript: Optional[dict] = None):
    """Yield a zip of scenes.json, scenes.csv and a 360p thumbnail per scene, streamed as the
    thumbnails are made (a few at a time; they're cached for the next export). With a
    transcript, its subtitles too."""
    proxy = root / "proxy.mp4"
    sink = _Sink()

    def thumb(row):
        path = root / "thumbs" / f"{int(row['thumbnail_time'] * 1000)}_full.jpg"
        return path if make_thumb(proxy, int(row["thumbnail_time"] * 1000), path, None) else None

    pool = ThreadPoolExecutor(max_workers=4)
    try:
        with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            zf.writestr(f"{folder_name}/scenes.json", json.dumps(doc, indent=2, ensure_ascii=False))
            zf.writestr(f"{folder_name}/scenes.csv", export_csv(doc))
            if transcript:
                zf.writestr(f"{folder_name}/transcript.srt", speech.to_srt(transcript["segments"]))
            yield from sink.drain()
            futures = [pool.submit(thumb, row) for row in doc["scenes"]]
            for row, future in zip(doc["scenes"], futures):
                path = future.result()
                if path:
                    zf.write(path, f"{folder_name}/{row['thumbnail']}")
                    yield from sink.drain()
        yield from sink.drain()
    finally:
        # The download may stop part-way: don't make the thumbnails nobody will receive
        pool.shutdown(wait=False, cancel_futures=True)


# ---------- retention ----------

def sweep() -> None:
    if not VIDEO_DIR.exists():
        return
    now = time.time()
    for root in VIDEO_DIR.iterdir():
        if not ID_RE.match(root.name) or not root.is_dir() or active_jobs(root.name):
            continue
        try:
            partial = root / "source.part"
            if partial.exists():
                stale = now - partial.stat().st_mtime > UPLOAD_RETENTION_DAYS * 86400
            else:
                stale = now - (root / "meta.json").stat().st_mtime > RETENTION_DAYS * 86400
        except OSError:
            stale = not (root / "meta.json").exists() and now - root.stat().st_mtime > 3600
        if stale:
            shutil.rmtree(root, ignore_errors=True)


# ---------- API ----------

@asynccontextmanager
async def lifespan(_app):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for tmp in VIDEO_DIR.glob("*/tmp"):
        shutil.rmtree(tmp, ignore_errors=True)  # left by a restart mid-job
    sweep()
    threading.Thread(target=worker, daemon=True, name="video-worker").start()
    yield


app = FastAPI(title="Video scenes", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "queued": _queue.qsize()}


@app.get("/detectors")
def detectors():
    return {
        "detectors": [
            {"id": d.id, "label": d.label, "description": d.description, "available": d.available, "note": d.note,
             "live": d.live, "curve_param": d.curve_param, "params": [p.__dict__ for p in d.params],
             # Which cut rule the browser's threshold preview must copy (see cutsFromScores in video.js)
             "rule": {"cuts_at_peaks": "peaks", "cuts_above": "above"}.get(getattr(d.rescore, "__name__", ""))}
            for d in DETECTORS.values()
        ]
    }


@app.get("/speech-models")
def speech_models():
    return {
        "models": [{"id": m.id, "label": m.label, "size_mb": m.size_mb, "note": m.note,
                    "downloaded": speech.downloaded(m)} for m in speech.MODELS.values()],
        "default": speech.DEFAULT_MODEL,
        "languages": speech.languages(),
        "diarization": {"label": diarize.LABEL, "available": diarize.available(), "downloaded": diarize.downloaded(),
                        "size_mb": diarize.SIZE_MB, "max_speakers": diarize.MAX_SPEAKERS, "note": diarize.note()},
    }


def status(video_id: str, root: Path, full: bool) -> dict:
    meta = read_json(root / "meta.json", {})
    partial = root / "source.part"
    src = source_path(root)
    received = partial.stat().st_size if partial.exists() else (src.stat().st_size if src else 0)
    info = read_json(root / "info.json")
    prep = ensure_prepared(video_id, root) if full else _latest(video_id, "prepare")
    if not src:
        state = "uploading"
    elif (root / "proxy.mp4").exists():
        state = "ready"
    elif prep and prep.state in ("error", "cancelled"):
        state = prep.state
    else:
        state = "processing"
    out = {
        "id": video_id, "name": meta.get("name"), "size": meta.get("size"), "received": received,
        "created": meta.get("created"), "state": state, "duration": (info or {}).get("duration"),
    }
    if full:
        out["info"] = info
        out["error"] = prep.error if prep and prep.state == "error" else None
        out["jobs"] = [j.public() for j in active_jobs(video_id)]
        with _jobs_lock:
            # Recently finished detections, so the page can tell a failure from a success
            out["recent"] = [j.public() for j in _jobs.get(video_id, {}).values() if j.state in ("error", "cancelled")]
        out["detections"] = sorted(
            filter(None, (summary(p) for p in (root / "scenes").glob("*.json"))), key=lambda s: s.get("created") or 0)
        out["transcripts"] = sorted(
            filter(None, (transcript_summary(p) for p in (root / "transcripts").glob("*.json"))),
            key=lambda s: s.get("created") or 0)
        out["speakers"] = sorted(
            filter(None, (speakers_summary(p) for p in (root / "speakers").glob("a*.json"))),
            key=lambda s: s["track"])
        out["subtitled"] = subtitled_list(root)
    return out


@app.get("/videos")
def list_videos():
    sweep()
    out = []
    for root in VIDEO_DIR.iterdir() if VIDEO_DIR.exists() else []:
        if ID_RE.match(root.name) and (root / "meta.json").is_file():
            out.append(status(root.name, root, full=False))
    return {"videos": sorted(out, key=lambda v: v.get("created") or 0, reverse=True)}


@app.get("/videos/{video_id}")
def get_video(video_id: str):
    root = folder(video_id)
    touch(root)
    return status(video_id, root, full=True)


@app.delete("/videos/{video_id}")
def delete_video(video_id: str):
    root = folder(video_id)
    cancel_jobs(video_id)
    shutil.rmtree(root, ignore_errors=True)
    with _jobs_lock:
        _jobs.pop(video_id, None)
    return {"deleted": video_id}


@app.post("/videos/{video_id}/cancel")
def cancel(video_id: str):
    folder(video_id)
    return {"cancelled": cancel_jobs(video_id)}


class DetectRequest(BaseModel):
    detector: str
    params: dict = {}


@app.post("/videos/{video_id}/detect")
def start_detect(video_id: str, req: DetectRequest):
    root = folder(video_id)
    detector = DETECTORS.get(req.detector)
    if not detector:
        raise HTTPException(404, f"Unknown detector '{req.detector}'.")
    if not detector.available:
        raise HTTPException(400, detector.note or f"{detector.label} is not available.")
    params = detector.normalize(req.params)
    key = detector.key(params)
    if (root / "scenes" / f"{key}.json").exists():
        return {"key": key, "cached": True}
    info = read_json(root / "info.json")
    scores = scores_path(root, detector.id)
    if detector.live and info and scores.exists():
        # Same scores at any threshold: cut them again rather than decoding the video again
        rescore(root, detector, params, key, info, np.fromfile(scores, "<f4"), 0)
        return {"key": key, "cached": True}
    if not source_path(root):
        raise HTTPException(409, "The upload is not complete.")
    if not (root / "proxy.mp4").exists():
        # Retry a failed or stopped preparation; the detection queues behind it
        prep = _latest(video_id, "prepare")
        if not prep or prep.state not in ("queued", "running"):
            submit(video_id, "prepare", "Preparing video", prepare)
    job = submit(video_id, "detect", detector.label, detect(detector.id, params), key=key)
    return {"key": key, "cached": False, "job": job.public()}


@app.get("/videos/{video_id}/scenes/{key}")
def get_scenes(video_id: str, key: str):
    return scene_result(folder(video_id) / "scenes" / f"{check_key(key)}.json")


@app.get("/videos/{video_id}/scores/{detector_id}")
def get_scores(video_id: str, detector_id: str):
    """A live detector's per-frame scores as little-endian float32, for the threshold preview."""
    if detector_id not in DETECTORS:
        raise HTTPException(404, "Unknown detector.")
    path = scores_path(folder(video_id), detector_id)
    if not path.exists():
        raise HTTPException(404, "No scores saved for this detector; run it once.")
    return FileResponse(path, media_type="application/octet-stream")


@app.delete("/videos/{video_id}/scenes/{key}")
def delete_scenes(video_id: str, key: str):
    root = folder(video_id)
    (root / "scenes" / f"{check_key(key)}.json").unlink(missing_ok=True)
    shutil.rmtree(root / "tags" / key, ignore_errors=True)  # its tags describe scenes that are gone
    return {"deleted": key}


@app.get("/videos/{video_id}/scenes/{key}/tags")
def get_tags(video_id: str, key: str, config: str = ""):
    """Which models and workflows have tagged this result; with config, that one's saved entries
    by scene index (without the run records)."""
    root = folder(video_id)
    out = {"configs": tag_configs(root, check_key(key))}
    if config:
        out["scenes"] = tag_entries(tags_dir(root, key, config))
    return out


@app.get("/videos/{video_id}/scenes/{key}/tags/{scene}")
def get_tag(video_id: str, key: str, scene: int, config: str):
    """One scene's entry with the full run that made it."""
    path = tags_dir(folder(video_id), check_key(key), config)
    entry = read_json(path / f"{scene}.json")
    if entry is None:
        raise HTTPException(404, "This scene isn't tagged yet.")
    return {**entry, "record": read_json(path / f"{scene}.record.json", {})}


class TagSave(BaseModel):
    label: str = ""  # the config as the page shows it, e.g. "workflow: Catalog"
    ok: bool
    result: Any = None  # the JSON the model or workflow returned
    error: Optional[str] = None
    record: dict = {}  # the whole run, so the page can show it again


@app.put("/videos/{video_id}/scenes/{key}/tags/{scene}")
def put_tag(video_id: str, key: str, scene: int, config: str, req: TagSave):
    root = folder(video_id)
    scenes = scene_result(root / "scenes" / f"{check_key(key)}.json")["scenes"]
    if not 0 <= scene < len(scenes):
        raise HTTPException(404, "No such scene.")
    if not config.strip():
        raise HTTPException(400, "Say which model or workflow made these tags.")
    path = tags_dir(root, key, config)
    path.mkdir(parents=True, exist_ok=True)
    label = req.label or config
    if (read_json(path / "config.json") or {}).get("label") != label:
        write_json(path / "config.json", {"config": config, "label": label})
    sc = scenes[scene]
    # The scene's times go in with its tags, so the files make sense on their own
    entry = {"scene": scene, "start": sc["start"], "end": sc["end"], "ok": req.ok, "result": req.result,
             "error": req.error, "source": label, "at": time.time()}
    write_json(path / f"{scene}.record.json", req.record)
    write_json(path / f"{scene}.json", entry)
    return entry


@app.delete("/videos/{video_id}/scenes/{key}/tags")
def delete_tags(video_id: str, key: str, config: str):
    shutil.rmtree(tags_dir(folder(video_id), check_key(key), config), ignore_errors=True)
    return {"deleted": config}


class TranscribeRequest(BaseModel):
    model: str = speech.DEFAULT_MODEL
    language: str = ""  # empty: detect it
    track: int = 0  # which audio track, counting from 0


@app.post("/videos/{video_id}/transcribe")
def start_transcribe(video_id: str, req: TranscribeRequest):
    root = folder(video_id)
    info = read_json(root / "info.json")
    if info and not info["audio"]:
        raise HTTPException(400, "This video has no audio track.")
    try:
        params = speech.normalize(req.model_dump(), len((info or {}).get("audio") or [None]))
    except ValueError as e:
        raise HTTPException(400, str(e))
    key = speech.key(params)
    if transcript_path(root, key).exists():
        return {"key": key, "cached": True}
    if not source_path(root):
        raise HTTPException(409, "The upload is not complete.")
    if not (root / "proxy.mp4").exists():
        prep = _latest(video_id, "prepare")
        if not prep or prep.state not in ("queued", "running"):
            submit(video_id, "prepare", "Preparing video", prepare)
    job = submit(video_id, "transcribe", speech.label(params), run_transcribe(params), key=key)
    return {"key": key, "cached": False, "job": job.public()}


def transcript_result(root: Path, key: str) -> dict:
    data = read_json(transcript_path(root, check_key(key)))
    if data is None:
        raise HTTPException(404, "No such transcript.")
    return data


@app.get("/videos/{video_id}/transcripts/{key}")
def get_transcript(video_id: str, key: str):
    root = folder(video_id)
    return with_speakers(root, transcript_result(root, key))


@app.delete("/videos/{video_id}/transcripts/{key}")
def delete_transcript(video_id: str, key: str):
    root = folder(video_id)
    transcript_path(root, check_key(key)).unlink(missing_ok=True)
    drop_subtitled(root, [key])
    return {"deleted": key}


class DiarizeRequest(BaseModel):
    track: int = 0  # which audio track, counting from 0
    speakers: int = 0  # how many speakers there are; 0: find out


@app.post("/videos/{video_id}/diarize")
def start_diarize(video_id: str, req: DiarizeRequest):
    root = folder(video_id)
    info = read_json(root / "info.json")
    if info and not info["audio"]:
        raise HTTPException(400, "This video has no audio track.")
    if not diarize.available():
        raise HTTPException(400, diarize.note())
    params = diarize.normalize(req.model_dump(), len((info or {}).get("audio") or [None]))
    key = diarize.key(params)
    done = read_json(speakers_path(root, params["track"]))
    # A result from another model (3.1, before community-1) runs again
    if done and done.get("params") == params and done.get("model") == diarize.MODEL:
        return {"key": key, "cached": True}
    if not source_path(root):
        raise HTTPException(409, "The upload is not complete.")
    if not (root / "proxy.mp4").exists():
        prep = _latest(video_id, "prepare")
        if not prep or prep.state not in ("queued", "running"):
            submit(video_id, "prepare", "Preparing video", prepare)
    job = submit(video_id, "diarize", "Speakers", run_diarize(params), key=key)
    return {"key": key, "cached": False, "job": job.public()}


@app.get("/videos/{video_id}/speakers/{track}")
def get_speakers(video_id: str, track: int):
    data = read_json(speakers_path(folder(video_id), track))
    if data is None:
        raise HTTPException(404, "No speakers for that track.")
    return data


@app.delete("/videos/{video_id}/speakers/{track}")
def delete_speakers(video_id: str, track: int):
    root = folder(video_id)
    speakers_path(root, track).unlink(missing_ok=True)
    drop_subtitled(root, transcript_keys(root, track))
    return {"deleted": track}


@app.get("/videos/{video_id}/transcripts/{key}/video")
def subtitled_video(video_id: str, key: str):
    """The original video with the transcript as a subtitle track: its video and audio streams
    copied into Matroska, which takes any codec, streamed as ffmpeg writes it. Data streams
    (cameras' timecode tracks) and the source's own subtitles, which Matroska may not take as
    they are, are left out."""
    root = folder(video_id)
    data = with_speakers(root, transcript_result(root, key))
    info = read_json(root / "info.json")
    src = source_path(root)
    if not info or not src:
        raise HTTPException(409, "The video isn't ready yet.")
    srt = write_srt(root, data)
    lang = speech.iso639_2(data.get("language"))
    title = data["label"] + (" with speakers" if data.get("speakers") else "")
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(src), "-i", str(srt),
         "-map", f"0:{info['video']['index']}", "-map", "0:a?", "-map", "1:0", "-c", "copy", "-c:s", "srt",
         "-metadata:s:s:0", f"language={lang}", "-metadata:s:s:0", f"title={title}",
         "-disposition:s:0", "default", "-f", "matroska", "pipe:1"],
        stdout=subprocess.PIPE, stderr=err,
    )

    def cleanup():
        srt.unlink(missing_ok=True)
        err.close()

    # The first bytes come quickly; if ffmpeg fails before them, say why instead of sending an empty file
    first = proc.stdout.read(1 << 16)
    if not first:
        code = proc.wait()
        err.seek(0)
        message = err.read().decode(errors="replace").strip()[-500:]
        cleanup()
        raise HTTPException(500, f"ffmpeg exited {code}: {message}")
    name = f"{_stem(root)}_{data['language'] or 'speech'}_subtitled.mkv"
    return StreamingResponse(mux_stream(proc, first, cleanup), media_type="video/x-matroska",
                             headers=_disposition(name))


class BurnRequest(BaseModel):
    size: str = "360"  # "360": the 360p preview copy, quick; "full": the original's size


@app.post("/videos/{video_id}/transcripts/{key}/burn")
def start_burn(video_id: str, key: str, req: BurnRequest):
    root = folder(video_id)
    transcript_result(root, key)
    if req.size not in BURN_SIZES:
        raise HTTPException(400, "Invalid size.")
    if subtitled_path(root, key, req.size).exists():
        return {"name": f"{key}-{req.size}", "cached": True}
    info = read_json(root / "info.json")
    proxy = root / "proxy.mp4"
    if not info or not proxy.exists():
        raise HTTPException(409, "The video isn't ready yet.")
    # About as big as what it's re-encoded from, at these settings
    need = info["size"] if req.size == "full" else proxy.stat().st_size * 2
    free = shutil.disk_usage(VIDEO_DIR).free
    if free < need + (1 << 30):
        raise HTTPException(507, f"Not enough disk space: burning in subtitles needs about {need / 1e9:.1f} GB "
                                 f"and the video volume has {free / 1e9:.1f} GB free.")
    label = "Subtitles · " + ("360p" if req.size == "360" else "original size")
    job = submit(video_id, "burn", label, run_burn(key, req.size), key=f"burn-{key}-{req.size}")
    return {"name": f"{key}-{req.size}", "cached": False, "job": job.public()}


def subtitled_file(root: Path, name: str) -> Path:
    key, _, size = name.rpartition("-")
    if not KEY_RE.match(name) or size not in BURN_SIZES:
        raise HTTPException(400, "Invalid name.")
    return subtitled_path(root, key, size)


@app.get("/videos/{video_id}/subtitled/{name}")
def get_subtitled(video_id: str, name: str):
    root = folder(video_id)
    path = subtitled_file(root, name)
    if not path.exists():
        raise HTTPException(404, "No such video.")
    key, _, size = name.rpartition("-")
    lang = (read_json(transcript_path(root, key)) or {}).get("language") or "speech"
    filename = f"{_stem(root)}_{lang}_subtitled{'_360p' if size == '360' else ''}.mp4"
    return FileResponse(path, media_type="video/mp4", headers=_disposition(filename))


@app.delete("/videos/{video_id}/subtitled/{name}")
def delete_subtitled(video_id: str, name: str):
    subtitled_file(folder(video_id), name).unlink(missing_ok=True)
    return {"deleted": name}


@app.get("/videos/{video_id}/transcripts/{key}/export")
def export_transcript(video_id: str, key: str, format: str = "srt"):
    """Subtitles (srt, vtt), plain text with a time per line (txt), or every segment and word (json)."""
    root = folder(video_id)
    data = with_speakers(root, transcript_result(root, key))
    name = f"{_stem(root)}_{data['language'] or 'speech'}"
    segments = data["segments"]
    if format == "srt":
        return Response(speech.to_srt(segments), media_type="application/x-subrip; charset=utf-8",
                        headers=_disposition(f"{name}.srt"))
    if format == "vtt":
        return Response(speech.to_vtt(segments), media_type="text/vtt; charset=utf-8",
                        headers=_disposition(f"{name}.vtt"))
    if format == "txt":
        return Response(speech.to_text(segments), media_type="text/plain; charset=utf-8",
                        headers=_disposition(f"{name}.txt"))
    if format == "json":
        meta = read_json(root / "meta.json", {})
        doc = {
            "format": "photo-organizer/video-transcript", "version": 1,
            "exported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "video": {"name": meta.get("name"), "duration": data["duration"]},
            "model": data["label"], "language": data["language"],
            "language_probability": data["language_probability"],
            **({"speakers": data["speakers"]} if data.get("speakers") else {}), "segments": segments,
        }
        return Response(json.dumps(doc, indent=2, ensure_ascii=False), media_type="application/json",
                        headers=_disposition(f"{name}.json"))
    raise HTTPException(400, "Invalid export format.")


@app.get("/videos/{video_id}/thumb")
def thumb(video_id: str, t: float):
    root = folder(video_id)
    proxy = root / "proxy.mp4"
    if not proxy.exists():
        raise HTTPException(409, "The video isn't ready yet.")
    ms = max(0, int(t * 1000))
    path = root / "thumbs" / f"{ms}.jpg"
    try:
        if not make_thumb(proxy, ms, path, 120):
            raise HTTPException(404, "No frame at that time.")  # past the end
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=604800, immutable"})


@app.get("/videos/{video_id}/frame")
def frame(video_id: str, t: float):
    """The original's frame at t, fitted inside FRAME_EDGE: what tagging sends to the models, much
    sharper than the 360p proxy. Not kept on disk: each is sent to a model once or twice."""
    root = folder(video_id)
    src = source_path(root)
    info = read_json(root / "info.json")
    if not src or not info:
        raise HTTPException(409, "The video isn't ready yet.")
    fit = f"scale=w='min({FRAME_EDGE},iw)':h='min({FRAME_EDGE},ih)':force_original_aspect_ratio=decrease"
    # -ss before -i seeks to the keyframe before t, then decodes up to t exactly
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-ss", f"{min(max(t, 0), info['duration']):.3f}",
         "-i", str(src), "-map", f"0:{info['video']['index']}", "-frames:v", "1", "-vf", fit,
         "-q:v", "3", "-f", "image2pipe", "-c:v", "mjpeg", "pipe:1"],
        capture_output=True, timeout=120,
    )
    if out.returncode != 0:
        raise HTTPException(500, f"ffmpeg exited {out.returncode}: {out.stderr.decode(errors='replace').strip()[-500:]}")
    if not out.stdout:
        raise HTTPException(404, "No frame at that time.")  # past the end
    return Response(out.stdout, media_type="image/jpeg", headers={"Cache-Control": "max-age=604800, immutable"})


def _stem(root: Path) -> str:
    name = read_json(root / "meta.json", {}).get("name") or "video"
    stem = re.sub(r"[^\w.-]+", "_", Path(name).stem).strip("._") or "video"
    return stem[:80]


def _disposition(filename: str) -> dict:
    from urllib.parse import quote

    return {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"}


@app.get("/videos/{video_id}/export")
def export(video_id: str, key: str, format: str = "zip", config: str = "", transcript: str = ""):
    """A result's scene list: json or csv, or a zip of both with a thumbnail per scene. With
    config, scenes that model or workflow tagged include their JSON; with transcript (a key),
    each scene includes what was said in it."""
    root = folder(video_id)
    if format not in ("zip", "json", "csv") or not KEY_RE.match(key):
        raise HTTPException(400, "Invalid export request.")
    info = read_json(root / "info.json")
    if not info or not (root / "proxy.mp4").exists():
        raise HTTPException(409, "The video isn't ready yet.")
    result = scene_result(root / "scenes" / f"{key}.json")
    name = f"{_stem(root)}_{result['detector']}_scenes"
    spoken = with_speakers(root, transcript_result(root, transcript)) if transcript else None
    doc = export_doc(root, info, result, thumbnails=format == "zip", config=config, transcript=spoken)
    if format == "json":
        return Response(json.dumps(doc, indent=2, ensure_ascii=False), media_type="application/json",
                        headers=_disposition(f"{name}.json"))
    if format == "csv":
        return Response(export_csv(doc), media_type="text/csv; charset=utf-8", headers=_disposition(f"{name}.csv"))
    return StreamingResponse(export_zip(root, doc, name, spoken), media_type="application/zip",
                             headers=_disposition(f"{name}.zip"))
