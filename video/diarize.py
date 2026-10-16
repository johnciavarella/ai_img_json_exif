"""Who speaks when, with pyannote.audio's speaker-diarization-community-1 pipeline on the CPU.

The pipeline finds speech in overlapping 10 s windows, turns each window's voices into
WeSpeaker embeddings and clusters them into speakers across the whole recording (VBx). It reads
the same 16 kHz PCM that transcription does. The result is one list of turns per audio track,
independent of any transcript, taken from the pipeline's exclusive diarization (one speaker at
a time), which lines up with transcribed words: assign() gives each word the speaker whose turn
it overlaps and splits lines where the speaker changes, so every transcript of that track shows
speakers without diarizing again.

The pipeline is gated on Hugging Face (accept the terms on its page) and downloads once with
HF_TOKEN. Its repo carries the models it uses, so the files are fetched at a pinned commit and
the pipeline loads from that snapshot without the network.
"""

import gc
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

CACHE_DIR = Path(os.environ.get("VIDEO_CACHE", "/cache"))
SAMPLE_RATE = 16000
MAX_SPEAKERS = 10
NEAREST = 1.0  # seconds: a word outside every turn takes the nearest turn this close
MODEL = "speaker-diarization-community-1"
LABEL = f"pyannote {MODEL}"

# Pinned, so a changed upload can't change results
REPO = f"pyannote/{MODEL}"
REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
FILES = ("config.yaml", "segmentation/pytorch_model.bin", "embedding/pytorch_model.bin",
         "plda/plda.npz", "plda/xvec_transform.npz")
SIZE_MB = 33
GATED = (REPO,)

# Share of the progress bar per step; the embeddings take most of the time
STEPS = {"segmentation": (0.0, 0.2), "embeddings": (0.2, 0.97)}


def model_dir() -> Path:
    return CACHE_DIR / "pyannote"


def snapshot() -> Path:
    # The hub cache's layout: models--<org>--<name>/snapshots/<revision>/
    return model_dir() / f"models--{REPO.replace('/', '--')}" / "snapshots" / REVISION


def downloaded() -> bool:
    return all((snapshot() / f).exists() for f in FILES)


def token() -> str:
    return os.environ.get("HF_TOKEN", "").strip()


def available() -> bool:
    return downloaded() or bool(token())


def note() -> str:
    if downloaded():
        return "Finds who speaks when; every transcript of the track then shows the speakers."
    if token():
        return f"Downloads {SIZE_MB} MB on first use."
    return ("Set HF_TOKEN for the video service and accept the terms of "
            + " and ".join(f"https://hf.co/{r}" for r in GATED) + " to download the speaker models once.")


def normalize(params: Optional[dict], tracks: int) -> dict:
    params = params or {}
    try:
        track = int(params.get("track") or 0)
    except (TypeError, ValueError):
        track = 0
    try:
        speakers = int(params.get("speakers") or 0)
    except (TypeError, ValueError):
        speakers = 0
    return {"track": min(max(track, 0), max(tracks - 1, 0)), "speakers": min(max(speakers, 0), MAX_SPEAKERS)}


def key(params: dict) -> str:
    return f"speakers-a{params['track']}"


# ---------- diarization ----------

def fetch() -> Path:
    """The pinned pipeline folder on disk; downloads what's missing. Its config refers to the
    models as $model/<subfolder>, which pyannote resolves inside this folder."""
    if not downloaded():
        from huggingface_hub import hf_hub_download

        for filename in FILES:
            hf_hub_download(REPO, filename, revision=REVISION, cache_dir=str(model_dir()), token=token() or None)
    return snapshot()


def diarize(pcm_path: Path, params: dict, report: Callable[[float], None],
            stage: Callable[[str], None]) -> dict:
    """Diarize 16 kHz mono s16le PCM. report(fraction) raises once the job is cancelled, which
    also stops the pipeline: it's called from the pipeline's progress hook."""
    import torch
    from pyannote.audio import Pipeline

    stage("Loading the speaker models" if downloaded() else f"Downloading the speaker models ({SIZE_MB} MB, first use only)")
    pipeline = Pipeline.from_pretrained(str(fetch()))
    stage("")

    def hook(step, _artefact, file=None, total=None, completed=None):
        lo, hi = STEPS.get(step, (None, None))
        if lo is not None and total:
            report(lo + (hi - lo) * min(completed or 0, total) / total)
        elif step == "discrete_diarization":
            report(STEPS["embeddings"][1])

    try:
        report(0.0)
        # The whole track as float32: about 1 GB for 4 hours, as the pipeline clusters speakers
        # across all of it at once
        pcm = np.memmap(pcm_path, dtype="<i2", mode="r")
        waveform = torch.from_numpy(np.asarray(pcm, dtype=np.float32) / 32768.0)[None]
        del pcm
        found = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, hook=hook,
                         **({"num_speakers": params["speakers"]} if params["speakers"] else {}))
        del waveform
        # One speaker at a time, so each transcribed word falls in a single speaker's turn
        found = found.exclusive_speaker_diarization
        # Speakers numbered from 1 in the order they first speak
        numbers = {}
        turns = []
        for seg, _, label in sorted(found.itertracks(yield_label=True), key=lambda x: x[0].start):
            n = numbers.setdefault(label, len(numbers) + 1)
            turns.append({"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "speaker": n})
        return {"count": len(numbers), "turns": turns}
    finally:
        del pipeline
        gc.collect()


# ---------- reading ----------

def bands(turns: list, gap: float = 1.0) -> list:
    """[start, end, speaker] per stretch of one speaker, merging turns of the same speaker less
    than gap apart: the timeline's row."""
    out = []
    for t in turns:
        if out and out[-1][2] == t["speaker"] and t["start"] - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], round(t["end"], 1))
        else:
            out.append([round(t["start"], 1), round(t["end"], 1), t["speaker"]])
    return out


def _speakers_of(items: list, turns: list) -> list:
    """The speaker of each (start, end) in items, which must be in time order: the turn it
    overlaps most, else the nearest within NEAREST seconds, else None."""
    out = []
    active = []  # turns started before the item ends that may still overlap later items
    i = 0
    prev_end = None  # the latest turn known to have ended
    for start, end in items:
        while i < len(turns) and turns[i]["start"] < end:
            active.append(turns[i])
            i += 1
        keep = []
        for t in active:
            if t["end"] <= start:
                if prev_end is None or t["end"] > prev_end["end"]:
                    prev_end = t
            else:
                keep.append(t)
        active = keep
        best, overlap = None, 0.0
        for t in active:
            o = min(end, t["end"]) - max(start, t["start"])
            if o > overlap:
                best, overlap = t["speaker"], o
        if best is None:
            near = []
            if prev_end is not None:
                near.append((start - prev_end["end"], prev_end["speaker"]))
            if i < len(turns):
                near.append((turns[i]["start"] - end, turns[i]["speaker"]))
            near = [x for x in near if x[0] <= NEAREST]
            best = min(near)[1] if near else None
        out.append(best)
    return out


def assign(segments: list, turns: list) -> list:
    """The transcript's segments with a "speaker" each, split wherever the speaker changes
    between words."""
    if not turns:
        return segments
    items = []
    for s in segments:
        for w in s.get("words") or [s]:
            items.append((w["start"], w["end"]))
    order = sorted(range(len(items)), key=lambda k: items[k][0])
    found = _speakers_of([items[k] for k in order], turns)
    speakers = [None] * len(items)
    for k, sp in zip(order, found):
        speakers[k] = sp
    out = []
    n = 0
    for s in segments:
        words = s.get("words") or []
        if not words:
            out.append({**s, "speaker": speakers[n]})
            n += 1
            continue
        who = speakers[n:n + len(words)]
        n += len(words)
        # A word outside every turn joins the words around it
        for k in range(1, len(who)):
            if who[k] is None:
                who[k] = who[k - 1]
        for k in range(len(who) - 2, -1, -1):
            if who[k] is None:
                who[k] = who[k + 1]
        groups = []
        for w, sp in zip(words, who):
            if groups and groups[-1][0] == sp:
                groups[-1][1].append(w)
            else:
                groups.append((sp, [w]))
        for g, (sp, ws) in enumerate(groups):
            text = "".join(w["word"] for w in ws).strip()
            if not text:
                continue
            out.append({
                "start": s["start"] if g == 0 else ws[0]["start"],
                "end": s["end"] if g == len(groups) - 1 else ws[-1]["end"],
                "text": text, "words": ws, "speaker": sp,
            })
    return out

