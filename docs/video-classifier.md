# Video Scene Detection & Transcription Deep Dive

The Video Classifier splits videos of any duration into discrete scenes, transcribes speech with Whisper, identifies speakers, classifies individual scenes, and exports rich multimedia packages.

## Scene Detectors

| Detector | Finds | Characteristics & Recommendations |
|---|---|---|
| **TransNetV2** | Hard cuts, dissolves, fades | Deep neural network. Highest overall boundary accuracy. Generates per-frame scores. |
| **PySceneDetect · content** | Hard cuts | Very fast; lower the threshold slider to catch subtler cuts. |
| **PySceneDetect · adaptive** | Hard cuts | Adaptive rolling window; minimizes false cuts during fast camera pans or action. |
| **PySceneDetect · fades** | Fades to / from black | Detects lighting shifts where video fades out/in. |
| **FFmpeg scene filter** | Hard cuts | Fast built-in filter; try a threshold near `0.2` for balanced sensitivity. |

### Interactive Threshold Tuning
- TransNetV2 and FFmpeg precompute frame scores during the initial pass. Dragging the dashed threshold line updates scenes instantly in the browser without re-reading the video file.
- PySceneDetect recalculates scene cuts upon releasing the slider.

---

## Audio Transcription (Whisper)

Transcription is powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with Silero VAD (voice activity detection) running on the CPU:

- **Chunked processing**: Long videos are processed in 10-minute segments split cleanly at natural audio pauses, skipping silence.
- **Model selection**:
  - `tiny` / `base`: Fast and lightweight (~78 MB to ~145 MB).
  - `small`: Recommended balance of speed and accuracy (~488 MB).
  - `medium`: High accuracy (~1.5 GB).
  - `large-v3 turbo`: Excellent accuracy with enhanced speed compared to standard large (~1.6 GB).
  - `large-v3`: Maximum transcription fidelity (~3.1 GB).
- **Subtitles & Export Formats**:
  - **SRT / VTT**: Subtitle files split naturally between spoken words (max 2 lines, 6 seconds).
  - **TXT**: Plaintext with start timecode on each line.
  - **JSON**: Full hierarchical transcript with token-level timestamps and confidence scores.

---

## Speaker Diarization (pyannote)

**Identify speakers** finds who speaks when, using [pyannote.audio](https://github.com/pyannote/pyannote-audio) `speaker-diarization-community-1` on the CPU.

- **Setup (once):** the pipeline is gated on Hugging Face. Accept the terms of [`pyannote/speaker-diarization-community-1`](https://hf.co/pyannote/speaker-diarization-community-1), create a read token at <https://hf.co/settings/tokens>, put `HF_TOKEN=hf_...` in `.env`, and restart the video service (`podman compose up -d video`). The 33 MB pipeline downloads into `video-cache` on first use, pinned to a fixed revision; after that the token is no longer needed. Installs that used the earlier `speaker-diarization-3.1` need the new terms accepted and the token set once more. Until it is set, the button is disabled and its tooltip says why.
- **One result per audio track,** independent of any transcript: every Whisper transcript of that track shows the speakers, so you can re-transcribe with another model without identifying speakers again.
- **Turns:** one speaker at a time (the pipeline's exclusive diarization), so each word belongs to a single speaker. Results made with 3.1 stay until you identify speakers again.
- **Speaker count:** *Auto* detects it; choosing 1-10 fixes it, which usually helps when you know it. Running again with a different number replaces the result.
- **In the UI:** lines are split where the speaker changes, and a coloured **S1**, **S2**... label marks each change. A **Speakers** row under the timeline shows each speaker's stretches in their own colour (× deletes it), and the timeline's hover names the speaker. Speakers are numbered by who speaks first and cannot be renamed.
- **Exports:** SRT prefixes `Speaker N:` where the speaker changes, VTT uses voice tags (`<v Speaker N>`), TXT puts it on each line, and JSON adds `speaker` to each segment and a top-level `speakers` count. Scene `speech` text stays plain.
- **Memory:** the whole audio track is loaded as float32 (about 1 GB for 4 hours).
- **Licences:** pyannote.audio is MIT; the `speaker-diarization-community-1` pipeline and its models (segmentation, WeSpeaker ResNet34 embeddings, PLDA) are CC-BY-4.0.
- **Telemetry:** pyannote.audio 4 sends usage metrics by default; the video image turns this off (`PYANNOTE_METRICS_ENABLED=false`).

---

## Video with Subtitles

The **Video** button in the Transcript panel downloads the video with the transcript shown (with speakers, if identified) as subtitles:

| Option | How | Notes |
|---|---|---|
| **Subtitle track** | The original video and audio are stream-copied into a Matroska (`.mkv`) file with the subtitles as a track that players can switch on and off, tagged with the transcript's language. | Starts immediately, keeps full quality, needs no extra disk. Because it is streamed as it is made, the file has no duration in its header, so players may show the length as unknown and seek more slowly. Data tracks (such as a GoPro's timecode) are left out. |
| **Burned in, 360p** | A queued job draws the text onto the 360p preview copy (MP4). | About 30x faster than real time on an 8-core Apple Silicon Mac. |
| **Burned in, original size** | A queued job re-encodes the original (H.264, CRF 20, AAC) with the text drawn on (MP4). | About 10x faster than real time for 1080p; needs up to as much free disk as the original (the request is refused if space is short). Chinese, Japanese, and Korean text uses Noto CJK. |

Finished burned-in videos are listed in the dialog to download again or delete. They are removed when their transcript or the audio track's speakers are deleted or re-run, and with the video.

---

## Tagging Video Scenes

Once scenes are detected, open the **Tag scenes** tab:
1. Select a vision model, multimodal LLM, or multi-step workflow.
2. The server extracts the middle frame of each scene at up to 1536 px and runs the workflow.
3. Templates support video EXIF variables:
   - `{{exif.FileName}}`
   - `{{exif.Scene}}`
   - `{{exif.SceneStart}}`, `{{exif.SceneEnd}}`, `{{exif.SceneDuration}}`
   - `{{exif.Speech}}`: The transcribed spoken words that occurred during that specific scene.

---

## Performance & Disk Space Considerations

- **Disk Space**: Uploads require the video's original file size plus roughly 5–15% for the 360p preview copy. If using Docker Desktop or Podman with limited virtual disk sizes, mount a host directory using `VIDEO_DATA=/Volumes/Media/video-data`.
- **Retention**: Uploaded files and cached scenes are automatically pruned after `VIDEO_RETENTION_DAYS` (default 7 days) from the last access, or manually via the **delete** button.
- **Processing Speeds (Apple Silicon reference)**:
  - TransNetV2: ~90 fps on CPU (~1 hour for a 4-hour 25 fps video).
  - PySceneDetect / FFmpeg: Runs against the 360p proxy video (substantially faster).
  - Whisper `small`: Transcribes at ~5× real-time speed.
  - Burning in subtitles: ~30x real time at 360p and ~10x at 1080p (measured on an 8-core Apple Silicon Mac).
  - Speaker diarization: runs on the CPU as a queued job with a progress bar; it has not been timed on long recordings yet.

