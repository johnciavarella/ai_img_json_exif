# Command Line Interface (CLI)

The CLI tool (`python -m app.cli`) allows running classification workflows, generating vector embeddings, and tagging image collections directly from the terminal without opening a browser.

## Installation

The CLI needs Python 3.10 or newer. On macOS the built-in `python3` is 3.9, so install a current one first (`brew install python`) and make sure `python3 --version` shows it.

Create a dedicated virtual environment and install the CLI requirements:

```sh
python3 -m venv .venv-cli
.venv-cli/bin/pip install -r requirements-cli.txt
```

If you plan to write EXIF/IPTC/XMP tags directly into image copies (`--tag` or `--zip`), make sure [ExifTool](https://exiftool.org/) is installed and on your `$PATH`:
- **macOS**: `brew install exiftool`
- **Debian / Ubuntu**: `sudo apt install libimage-exiftool-perl`

Make sure the vision service is running if using embeddings or non-LLM classifiers (default port `127.0.0.1:8000`):
```sh
docker compose up -d vision
```

---

## Basic Usage

Export your workflow or prompt settings from the web UI via **Export settings** (e.g. `catalog.settings.json`), then pass it along with the path to your photo directory:

```sh
.venv-cli/bin/python -m app.cli catalog.settings.json ~/Pictures/Trip --tag --zip
```

---

## Command Flags

| Flag | Description |
|---|---|
| *(default)* | Emits `results.jsonl` (one JSON record per photo); safe to resume interrupted runs. |
| `--tag` | Generates non-destructive tagged copies preserving directory structures, plus `_run.json`. |
| `--zip` | Produces an archived zip file containing tagged copies, matching the Batch UI export. |
| `--sidecar` | Writes individual `<filename>.json` metadata sidecar files next to each photo. |
| `--embed` | Produces `embeddings.jsonl` containing 768-dimensional SigLIP 2 vectors for each photo. |
| `--like "TEXT"` | Natural-language semantic search ranking photos by descriptive relevance (best first). |
| `--overwrite` | Overwrites existing image metadata tags rather than preserving existing values. |
| `-o`, `--output` | Destination directory for output files (defaults to `photo-organizer-runs/<name>`). |
| `--rerun` | Forces re-evaluation of previously processed photos instead of skipping them. |
| `--api-key` | Provides an API key for the LLM backend (or set via `LLM_API_KEY` env var). |
| `--help` | Displays the complete CLI manual and option arguments. |

---

## Semantic Search & Vectors

### Generating Embeddings
Run `--embed` with or without a settings JSON:
```sh
.venv-cli/bin/python -m app.cli --embed ~/Pictures/Trip -o runs/trip
```
Each entry in `embeddings.jsonl` contains the photo's relative path and a normalized 768-float vector suitable for cosine similarity.

### Searching by Description
Search for images matching a prompt without running a heavy LLM:
```sh
.venv-cli/bin/python -m app.cli --like "sunset over water" ~/Pictures/Trip -o runs/trip
```
Pass the same output directory (`-o`) across queries to reuse precomputed embeddings so only new photos need to be embedded.

