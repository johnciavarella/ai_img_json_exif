"""Speech to text with faster-whisper (Whisper on CTranslate2, int8 on the CPU).

The audio track is decoded once to 16 kHz mono PCM next to the video and read through a memory
map, then transcribed in windows of about ten minutes, each ending at the quietest moment near
its end so no word is split. Given the whole file at once, faster-whisper would build its
spectrogram in one piece: several GB for a 4-hour video. Silero VAD (part of faster-whisper)
skips the silence inside each window, which also keeps Whisper from inventing text there.

Every segment keeps its words with their own times, so subtitles can be split at word
boundaries and each scene of an export gets exactly the words spoken in it.
"""

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

CACHE_DIR = Path(os.environ.get("VIDEO_CACHE", "/cache"))
SAMPLE_RATE = 16000
WINDOW = 600  # seconds of audio per transcribe call
QUIET_SEARCH = 30  # seconds before a window's end searched for the quietest moment
DEFAULT_MODEL = "small"


@dataclass
class Model:
    id: str
    label: str
    repo: str
    revision: str  # pinned, so a changed upload can't change results
    size_mb: int
    note: str


MODELS = {m.id: m for m in [
    Model("tiny", "Whisper tiny", "Systran/faster-whisper-tiny",
          "d90ca5fe260221311c53c58e660288d3deb8d356", 78, "Fastest; rough on accents and noise."),
    Model("base", "Whisper base", "Systran/faster-whisper-base",
          "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66", 148, "Fast; fine for clear speech."),
    Model("small", "Whisper small", "Systran/faster-whisper-small",
          "536b0662742c02347bc0e980a01041f333bce120", 486, "Good balance: about 5× faster than real time on 8 cores."),
    Model("medium", "Whisper medium", "Systran/faster-whisper-medium",
          "08e178d48790749d25932bbc082711ddcfdfbc4f", 1531, "More accurate than small; slower than large-v3 turbo."),
    Model("large-v3-turbo", "Whisper large-v3 turbo", "dropbox-dash/faster-whisper-large-v3-turbo",
          "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf", 1620, "Close to large-v3's accuracy; about 3× real time on 8 cores."),
    Model("large-v3", "Whisper large-v3", "Systran/faster-whisper-large-v3",
          "edaa852ec7e145841d8ffdb056a99866b5f0a478", 3091, "Most accurate; several times slower than turbo."),
]}


def languages() -> list:
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    return sorted(_LANGUAGE_CODES)


def model_dir() -> Path:
    return CACHE_DIR / "whisper"


def downloaded(model: Model) -> bool:
    # The hub cache's layout: models--<org>--<name>/snapshots/<revision>/
    snap = model_dir() / f"models--{model.repo.replace('/', '--')}" / "snapshots" / model.revision
    return (snap / "model.bin").exists()


def normalize(params: Optional[dict], tracks: int) -> dict:
    params = params or {}
    model = params.get("model") if params.get("model") in MODELS else DEFAULT_MODEL
    language = str(params.get("language") or "").lower()
    if language and language not in languages():
        raise ValueError(f"Whisper doesn't know the language '{language}'.")
    try:
        track = int(params.get("track") or 0)
    except (TypeError, ValueError):
        track = 0
    return {"model": model, "language": language, "track": min(max(track, 0), max(tracks - 1, 0))}


def key(params: dict) -> str:
    out = f"whisper-{params['model']}-{params['language'] or 'auto'}"
    return out + (f"-a{params['track']}" if params["track"] else "")


def label(params: dict) -> str:
    return MODELS[params["model"]].label


# ---------- transcription ----------

def sec(x) -> float:
    return round(float(x), 3)  # faster-whisper's times can be numpy floats


def quiet_point(pcm: np.ndarray, lo: int, hi: int) -> int:
    """The sample at the middle of the quietest 100 ms between lo and hi."""
    step = SAMPLE_RATE // 10
    chunk = pcm[lo:hi].astype(np.float32)
    n = len(chunk) // step
    if n < 2:
        return hi
    energy = np.square(chunk[: n * step].reshape(n, step)).mean(axis=1)
    return lo + int(np.argmin(energy)) * step + step // 2


def transcribe(pcm_path: Path, params: dict, report: Callable[[float], None],
               stage: Callable[[str], None], threads: int) -> dict:
    """Transcribe 16 kHz mono s16le PCM. report(fraction) raises once the job is cancelled."""
    from faster_whisper import WhisperModel

    m = MODELS[params["model"]]
    stage(f"Loading {m.label}" if downloaded(m) else f"Downloading {m.label} ({m.size_mb} MB, first use only)")
    model = WhisperModel(m.repo, device="cpu", compute_type="int8", cpu_threads=threads,
                         download_root=str(model_dir()), revision=m.revision)
    stage("")
    try:
        pcm = np.memmap(pcm_path, dtype="<i2", mode="r")
        total = len(pcm)
        language = params["language"] or None
        probability = None
        segments = []
        start = 0
        while start < total:
            end = min(total, start + WINDOW * SAMPLE_RATE)
            if end < total:
                end = quiet_point(pcm, max(start, end - QUIET_SEARCH * SAMPLE_RATE), end)
            offset = start / SAMPLE_RATE
            report(start / total)
            audio = np.asarray(pcm[start:end], dtype=np.float32) / 32768.0
            parts, info = model.transcribe(
                audio, language=language, vad_filter=True, word_timestamps=True,
                # Greedy: as accurate as a beam of 5 on test speech, and no slower on the CPU
                beam_size=1,
                # Each segment on its own: a misheard line can't start a loop that repeats it
                condition_on_previous_text=False,
            )
            found = False
            for seg in parts:
                report((start + seg.end * SAMPLE_RATE) / total)
                text = seg.text.strip()
                if not text:
                    continue
                found = True
                segments.append({
                    "start": sec(seg.start + offset), "end": sec(seg.end + offset), "text": text,
                    "words": [
                        {"start": sec(w.start + offset), "end": sec(w.end + offset),
                         "word": w.word, "probability": sec(w.probability)}
                        for w in seg.words or []
                    ],
                })
            if found and language is None:
                # Detected from the first window with speech, then kept, so the language can't
                # flip between windows
                language, probability = info.language, round(info.language_probability, 3)
            del audio
            start = end
        return {"language": language, "language_probability": probability, "segments": segments}
    finally:
        del model
        gc.collect()


# ---------- reading a transcript ----------

def spans(segments: list, gap: float = 1.0) -> list:
    """Speech as [start, end] spans, merging segments less than gap apart: the timeline's row."""
    out = []
    for s in segments:
        if out and s["start"] - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], round(s["end"], 1))
        else:
            out.append([round(s["start"], 1), round(s["end"], 1)])
    return out


def words_of(segments: list) -> list:
    """Every word with its times; a segment without word times counts as one word."""
    out = []
    for s in segments:
        out.extend(s.get("words") or [{"start": s["start"], "end": s["end"], "word": " " + s["text"]}])
    return out


def speech_by_scene(segments: list, scenes: list) -> list:
    """The text spoken in each scene: each word goes to the scene its middle falls in."""
    texts = [[] for _ in scenes]
    starts = [sc["start"] for sc in scenes]
    for w in words_of(segments):
        mid = (w["start"] + w["end"]) / 2
        i = int(np.searchsorted(starts, mid, side="right")) - 1
        if 0 <= i < len(scenes):
            texts[i].append(w["word"])
    return ["".join(t).strip() for t in texts]


# ---------- subtitles ----------

CUE_CHARS = 84  # two lines of 42
CUE_SECONDS = 6.0
LINE_CHARS = 42


def cues(segments: list) -> list:
    """Subtitle cues of at most two lines and CUE_SECONDS, split at word boundaries, as
    (start, end, text, speaker); speaker is None without diarization."""
    out = []
    for s in segments:
        who = s.get("speaker")
        words = s.get("words") or []
        if not words:
            out.append((s["start"], s["end"], s["text"], who))
            continue
        cur = []
        for w in words:
            text = "".join(x["word"] for x in cur + [w]).strip()
            if cur and (len(text) > CUE_CHARS or w["end"] - cur[0]["start"] > CUE_SECONDS):
                out.append((cur[0]["start"], cur[-1]["end"], "".join(x["word"] for x in cur).strip(), who))
                cur = []
            cur.append(w)
        if cur:
            out.append((cur[0]["start"], cur[-1]["end"], "".join(x["word"] for x in cur).strip(), who))
    return [c for c in out if c[2]]


def speaker_name(n) -> str:
    return f"Speaker {n}"


def wrap(text: str) -> str:
    """Break a cue over two lines at the space nearest its middle."""
    if len(text) <= LINE_CHARS:
        return text
    mid = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text
    i = min(spaces, key=lambda x: abs(x - mid))
    return text[:i] + "\n" + text[i + 1:]


def stamp(sec: float, sep: str) -> str:
    ms = max(0, round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: list) -> str:
    """SRT, naming the speaker on the first cue after each change of speaker."""
    out = []
    last = None
    for i, (a, b, t, who) in enumerate(cues(segments), 1):
        if who is not None and who != last:
            t = f"{speaker_name(who)}: {t}"
        last = who
        out.append(f"{i}\n{stamp(a, ',')} --> {stamp(b, ',')}\n{wrap(t)}\n")
    return "\n".join(out)


def to_vtt(segments: list) -> str:
    """WebVTT, with a voice tag on each cue whose speaker is known."""
    def text(t, who):
        t = wrap(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<v {speaker_name(who)}>{t}" if who is not None else t

    return "WEBVTT\n\n" + "\n".join(f"{stamp(a, '.')} --> {stamp(b, '.')}\n{text(t, who)}\n"
                                       for a, b, t, who in cues(segments))


def to_text(segments: list) -> str:
    def who(s):
        return f"{speaker_name(s['speaker'])}: " if s.get("speaker") is not None else ""

    return "".join(f"[{stamp(s['start'], '.')[:-4]}] {who(s)}{s['text']}\n" for s in segments)


# ISO 639-2/B codes for Whisper's languages, which Matroska's language tag expects (Whisper
# gives ISO 639-1). Cantonese has no 639-2 code of its own, so it's tagged as Chinese.
ISO639_2 = dict(pair.split(":") for pair in """
af:afr am:amh ar:ara as:asm az:aze ba:bak be:bel bg:bul bn:ben bo:tib br:bre bs:bos ca:cat
cs:cze cy:wel da:dan de:ger el:gre en:eng es:spa et:est eu:baq fa:per fi:fin fo:fao fr:fre
gl:glg gu:guj ha:hau haw:haw he:heb hi:hin hr:hrv ht:hat hu:hun hy:arm id:ind is:ice it:ita
ja:jpn jw:jav ka:geo kk:kaz km:khm kn:kan ko:kor la:lat lb:ltz ln:lin lo:lao lt:lit lv:lav
mg:mlg mi:mao mk:mac ml:mal mn:mon mr:mar ms:may mt:mlt my:bur ne:nep nl:dut nn:nno no:nor
oc:oci pa:pan pl:pol ps:pus pt:por ro:rum ru:rus sa:san sd:snd si:sin sk:slo sl:slv sn:sna
so:som sq:alb sr:srp su:sun sv:swe sw:swa ta:tam te:tel tg:tgk th:tha tk:tuk tl:tgl tr:tur
tt:tat uk:ukr ur:urd uz:uzb vi:vie yi:yid yo:yor zh:chi yue:chi
""".split())


def iso639_2(code: Optional[str]) -> str:
    return ISO639_2.get(code or "", "und")
